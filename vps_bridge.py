#!/usr/bin/env python3
"""
VPS side — local proxy bridge.
Listens on 127.0.0.1:8888 (for the bot to use as HTTPS_PROXY).
Accepts a reverse connection from Mac on 0.0.0.0:8889.
When Mac connects, all proxy traffic tunnels through it.
"""
import socket
import threading
import sys
import select

LISTEN_PROXY = ("127.0.0.1", 8890)  # bot connects here
LISTEN_MAC = ("0.0.0.0", 8889)  # Mac connects here
MAC_CONN = None
MAC_LOCK = threading.Lock()

def bridge(src, dst, label):
    """Bidirectional copy between two sockets."""
    try:
        while True:
            r, _, _ = select.select([src, dst], [], [], 300)
            if not r:
                break
            for s in r:
                data = s.recv(65536)
                if not data:
                    return
                if s is src:
                    dst.sendall(data)
                else:
                    src.sendall(data)
    except Exception:
        pass

def handle_mac(conn):
    global MAC_CONN
    with MAC_LOCK:
        MAC_CONN = conn
    print("Mac connected.")
    try:
        # Keep alive — read with timeout
        while True:
            data = conn.recv(1)
            if not data:
                break
    except Exception:
        pass
    with MAC_LOCK:
        MAC_CONN = None
    print("Mac disconnected.")
    try:
        conn.close()
    except:
        pass

def handle_client(client_conn):
    """Handle one bot proxy connection through Mac tunnel."""
    with MAC_LOCK:
        mac = MAC_CONN
    if not mac:
        try:
            client_conn.close()
        except:
            pass
        return

    # Create a fresh connection to Mac for this proxy session
    try:
        mac_sock = socket.create_connection(("127.0.0.1", 8889))
    except Exception:
        try:
            client_conn.close()
        except:
            pass
        return

    t1 = threading.Thread(target=bridge, args=(client_conn, mac_sock, "c2m"), daemon=True)
    t2 = threading.Thread(target=bridge, args=(mac_sock, client_conn, "m2c"), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=300)
    t2.join(timeout=300)
    try:
        client_conn.close()
    except:
        pass
    try:
        mac_sock.close()
    except:
        pass

def accept_mac():
    """Accept the Mac's reverse connection."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(LISTEN_MAC)
    sock.listen(1)
    print(f"Waiting for Mac on {LISTEN_MAC[0]}:{LISTEN_MAC[1]}...")
    while True:
        conn, addr = sock.accept()
        print(f"Mac from {addr}")
        t = threading.Thread(target=handle_mac, args=(conn,), daemon=True)
        t.start()

def main():
    # Start Mac listener
    threading.Thread(target=accept_mac, daemon=True).start()

    # Start proxy listener for bot
    proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy.bind(LISTEN_PROXY)
    proxy.listen(50)
    print(f"Proxy bridge on {LISTEN_PROXY[0]}:{LISTEN_PROXY[1]}")
    print("Run on Mac: python3 mac_tunnel.py 100.67.155.37 8889")
    
    while True:
        conn, addr = proxy.accept()
        t = threading.Thread(target=handle_client, args=(conn,), daemon=True)
        t.start()

if __name__ == "__main__":
    main()
