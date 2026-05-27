#!/usr/bin/env python3
"""
polymarket_deposit.py — Direct on-chain USDC deposit to Polymarket CLOB.
Bypasses polymarket.com geo-block. Works from VPS.

Flow: USDC.approve(Onramp) → Onramp.wrap(USDC, wallet, amount)

Usage:
    python3 polymarket_deposit.py AMOUNT
    python3 polymarket_deposit.py 48.90  # deposit all
    python3 polymarket_deposit.py 10     # deposit $10
"""
import sys
import json
import os
from web3 import Web3
from eth_account import Account

# ===== CONFIG =====
POLYGON_RPC = os.environ.get("POLYGON_RPC", "https://1rpc.io/matic")

# USDC (native) on Polygon
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

# Polymarket CTF Exchange V2 contracts
COLLATERAL_ONRAMP = "0x93070a847efEf7F70739046A929D47a521F5B8ee"

# Also need to approve the exchanges for trading
NEG_RISK_EXCHANGE = "0xe2222d279d744050d28e00520010520000310F59"
CTF_EXCHANGE = "0xE111180000d2663C0091e4f400237545B87B996B"

# ===== ABIs =====
ERC20_ABI = json.loads('[{"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}]')

ONRAMP_ABI = json.loads('[{"constant":false,"inputs":[{"name":"_asset","type":"address"},{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],"name":"wrap","outputs":[],"type":"function"}]')


def get_key():
    """Get private key from env or config file."""
    key = os.environ.get("POLYMARKET_KEY")
    if key:
        return key
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        key = cfg.get("private_key")
        if key:
            return key
    
    print("ERROR: No private key found. Set POLYMARKET_KEY env var or add to config.json")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 polymarket_deposit.py AMOUNT_USDC")
        print("Example: python3 polymarket_deposit.py 48.90")
        sys.exit(1)
    
    amount = float(sys.argv[1])
    key = get_key()
    
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    acct = Account.from_key(key)
    wallet = acct.address
    
    print(f"Wallet:    {wallet}")
    print(f"Amount:    ${amount:.2f} USDC")
    print(f"RPC:       {POLYGON_RPC}")
    print()
    
    # Check balances
    usdc = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
    balance = usdc.functions.balanceOf(wallet).call() / 1e6
    pol = w3.eth.get_balance(wallet) / 1e18
    
    print(f"USDC balance: ${balance:.2f}")
    print(f"POL balance:   {pol:.4f}")
    print()
    
    if balance < amount:
        print(f"ERROR: Insufficient balance (have ${balance:.2f}, need ${amount:.2f})")
        sys.exit(1)
    
    amount_raw = int(amount * 1e6)
    
    # Step 1: Approve Onramp to pull USDC
    print("=" * 55)
    print("STEP 1: Approve CollateralOnramp for USDC")
    print("=" * 55)
    print(f"  Onramp: {COLLATERAL_ONRAMP}")
    
    allowance = usdc.functions.allowance(wallet, COLLATERAL_ONRAMP).call() / 1e6
    print(f"  Current allowance: ${allowance:.2f}")
    
    if allowance < amount:
        print(f"  Approving ${amount:.2f}...", end=" ", flush=True)
        tx = usdc.functions.approve(COLLATERAL_ONRAMP, amount_raw).build_transaction({
            'from': wallet,
            'nonce': w3.eth.get_transaction_count(wallet),
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"OK ({tx_hash.hex()[:10]}...)")
    else:
        print(f"  Allowance sufficient ✓")
    
    # Step 2: Wrap USDC → Collateral Token
    print()
    print("=" * 55)
    print("STEP 2: Wrap USDC → Polymarket Collateral")
    print("=" * 55)
    
    onramp = w3.eth.contract(address=COLLATERAL_ONRAMP, abi=ONRAMP_ABI)
    
    print(f"  Wrapping ${amount:.2f} USDC for {wallet[:10]}...")
    
    try:
        tx = onramp.functions.wrap(USDC_ADDRESS, wallet, amount_raw).build_transaction({
            'from': wallet,
            'nonce': w3.eth.get_transaction_count(wallet),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  TX: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            print(f"  DEPOSIT SUCCESS ✓")
        else:
            print(f"  DEPOSIT FAILED — transaction reverted")
            sys.exit(1)
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
    
    # Step 3: Approve exchanges for trading (optional but recommended)
    print()
    print("=" * 55)
    print("STEP 3: Approve Exchanges for Trading")
    print("=" * 55)
    
    for label, exchange in [
        ("CTF Exchange", CTF_EXCHANGE),
        ("Neg Risk Exchange", NEG_RISK_EXCHANGE),
    ]:
        allowance = usdc.functions.allowance(wallet, exchange).call() / 1e6
        if allowance < amount:
            print(f"  {label}: approving ${amount:.2f}...", end=" ", flush=True)
            try:
                tx = usdc.functions.approve(exchange, amount_raw).build_transaction({
                    'from': wallet,
                    'nonce': w3.eth.get_transaction_count(wallet),
                    'gas': 100000,
                    'gasPrice': w3.eth.gas_price,
                })
                signed = acct.sign_transaction(tx)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                w3.eth.wait_for_transaction_receipt(tx_hash)
                print(f"OK")
            except Exception as e:
                print(f"FAILED: {e}")
        else:
            print(f"  {label}: already approved ✓")
    
    print()
    print("=" * 55)
    print("DONE — funds should appear within 1-2 min")
    print("Verify: check bot balance or polymarket.com/portfolio")
    print("=" * 55)


if __name__ == "__main__":
    main()
