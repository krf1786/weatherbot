#!/usr/bin/env python3
"""
VPS proxy — single-socket, framed signal then raw bridge.
"""
import socket
import struct
import threading
import select

LISTEN = 8890
MAC_PORT = 8889
BUFSIZE = 65536

mac_conn = None
mac_lock = threading.Lock()
mac_alive = threading.Event()


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_frame(sock, data):
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_frame(sock):
    header = recv_exact(sock, 4)
    if not header:
        return None
    length = struct.unpack("!I", header)[0]
    if length > 10_000_000:
        return None
    return recv_exact(sock, length)


def bridge_raw(a, b):
    """Pure byte forwarding, no framing."""
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 300)
            if not r:
                break
            for s in r:
                data = s.recv(BUFSIZE)
                if not data:
                    return
                dst = b if s is a else a
                dst.sendall(data)
    except:
        pass


def handle_client(client_sock):
    data = b""
    while b"\r\n" not in data:
        chunk = client_sock.recv(4096)
        if not chunk:
            try: client_sock.close()
            except: pass
            return
        data += chunk
        if len(data) > 8192:
            try: client_sock.close()
            except: pass
            return

    line = data.split(b"\r\n")[0].decode()
    if not line.startswith("CONNECT "):
        try: client_sock.close()
        except: pass
        return

    target = line.split()[1]

    with mac_lock:
        conn = mac_conn

    if conn is None:
        try:
            client_sock.sendall(b"HTTP/1.1 502 No Mac\r\n\r\n")
            client_sock.close()
        except:
            pass
        return

    try:
        send_frame(conn, target.encode())
    except:
        _mac_lost(conn)
        try: client_sock.sendall(b"HTTP/1.1 502 Send Failed\r\n\r\n")
        except: pass
        try: client_sock.close()
        except: pass
        return

    resp = recv_frame(conn)
    if not resp or not resp.startswith(b"OK"):
        try:
            msg = resp or b"no response"
            client_sock.sendall(b"HTTP/1.1 502 Mac: " + msg + b"\r\n\r\n")
            client_sock.close()
        except:
            pass
        return

    try:
        client_sock.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
    except:
        return

    bridge_raw(client_sock, conn)
    try: client_sock.close()
    except: pass


def _mac_lost(conn):
    """Signal that the Mac connection is dead."""
    global mac_conn
    with mac_lock:
        if mac_conn is conn:
            mac_conn = None
    mac_alive.clear()
    try: conn.close()
    except: pass


def accept_mac():
    global mac_conn
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", MAC_PORT))
    sock.listen(1)
    print(f"Mac port {MAC_PORT}")
    while True:
        conn, addr = sock.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        mac_alive.set()
        with mac_lock:
            if mac_conn:
                try: mac_conn.close()
                except: pass
            mac_conn = conn
        print(f"Mac OK: {addr}")
        # Block until Mac disconnects. Periodically check if socket
        # is still alive (important when bridge_raw threads aren't using it).
        while True:
            mac_alive.wait(timeout=30)
            if not mac_alive.is_set():
                break  # _mac_lost was called
            # Check if socket actually died (crashed Mac without clean close)
            try:
                r, _, e = select.select([conn], [], [conn], 0)
                if e:
                    break
                if r:
                    # Socket readable — either data or EOF
                    d = conn.recv(1, socket.MSG_PEEK)
                    if not d:
                        break  # EOF
            except:
                break
        mac_alive.clear()
        with mac_lock:
            if mac_conn is conn:
                mac_conn = None
        print("Mac gone")
        try: conn.close()
        except: pass


def main():
    threading.Thread(target=accept_mac, daemon=True).start()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", LISTEN))
    s.listen(50)
    print(f"Proxy {LISTEN}")

    while True:
        c, _ = s.accept()
        threading.Thread(target=handle_client, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
