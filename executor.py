#!/usr/bin/env python3
"""
executor.py — Real Polymarket order execution layer for WeatherBet.
Wraps py-clob-client-v2 for order creation, cancellation, and status.

Requirements:
  - Private key of a Polygon wallet with USDC + allowances
  - py-clob-client-v2 installed

Usage:
  from executor import Executor, ExecutionMode
  
  exec = Executor(private_key="0x...", funder="0x...")
  order_id = exec.buy(token_id, price, size)
  exec.sell(token_id, price, size)
  exec.cancel(order_id)
  status = exec.get_order(order_id)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Lazy imports — CLOB SDK only loaded for live mode
# from py_clob_client_v2 import ClobClient
# from py_clob_client_v2.clob_types import OrderArgsV2, OrderType
# from eth_account import Account

# =============================================================================
# CONFIG
# =============================================================================

# Load from env or config
POLYMARKET_HOST = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
POLYMARKET_CHAIN_ID = int(os.environ.get("POLYMARKET_CHAIN_ID", 137))
USDC_DECIMALS = 10**6  # USDC on Polygon has 6 decimals

EXECUTOR_DATA_DIR = Path(__file__).parent / "data" / "executor"
EXECUTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EXECUTOR
# =============================================================================

class ExecutionMode:
    PAPER = "paper"
    LIVE  = "live"


class Executor:
    """
    Polymarket CLOB execution layer.
    
    Paper mode: tracks trades in local state (no chain interaction)
    Live mode:  submits real orders via CLOB API
    """
    
    def __init__(
        self,
        mode: str = ExecutionMode.PAPER,
        private_key: Optional[str] = None,
        funder: Optional[str] = None,
        fee_slippage: float = 0.0001,
    ):
        self.mode = mode
        self.private_key = private_key
        self.funder = funder
        self.fee_slippage = fee_slippage
        self._client = None
        self._wallet = None
        
        if mode == ExecutionMode.LIVE:
            self._init_live(private_key, funder)
    
    def _init_live(self, private_key: str, funder: str):
        """Initialize CLOB client with wallet credentials."""
        if not private_key:
            raise ValueError("private_key required for live mode")
        
        # Lazy imports — only load CLOB SDK when going live
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs
        from eth_account import Account
        
        # Derive wallet address from private key
        acct = Account.from_key(private_key)
        self._wallet = acct.address
        self._funder = funder or acct.address
        
        # Check if we have cached API credentials
        creds_file = EXECUTOR_DATA_DIR / "api_creds.json"
        creds = None
        if creds_file.exists():
            creds_dict = json.loads(creds_file.read_text())
            from py_clob_client.clob_types import ApiCreds
            creds = ApiCreds(**creds_dict)
        
        # Initialize client
        self._client = ClobClient(
            host=POLYMARKET_HOST,
            chain_id=POLYMARKET_CHAIN_ID,
            key=private_key,
            creds=creds,
            signature_type=2,  # EOA
            funder=self._funder,
        )
        
        # Get or create API credentials
        if not creds:
            try:
                api_creds = self._client.create_or_derive_api_creds()
                creds_file.write_text(json.dumps({
                    "api_key": api_creds.api_key,
                    "api_secret": api_creds.api_secret,
                    "api_passphrase": api_creds.api_passphrase,
                }, indent=2))
                self._client.set_api_creds(api_creds)
            except Exception as e:
                raise RuntimeError(f"Failed to create API credentials: {e}")
        
        # Verify connection
        try:
            self._client.get_ok()
        except Exception as e:
            raise RuntimeError(f"CLOB connection failed: {e}")
    
    def _live_client(self):
        if not self._client:
            raise RuntimeError("Executor not initialized in live mode")
        return self._client
    
    # ---- Balance & Status ----
    
    def get_balance(self) -> Dict[str, Any]:
        """Get USDC balance and allowances."""
        if self.mode == ExecutionMode.PAPER:
            return {"mode": "paper", "available": None}
        
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            bal = self._live_client().get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            return {
                "mode": "live",
                "wallet": self._wallet,
                "usdc_balance": float(bal.get("balance", 0)) / USDC_DECIMALS if bal.get("balance") else 0,
                "allowance": float(bal.get("allowance", 0)) / USDC_DECIMALS if bal.get("allowance") else 0,
                "raw": bal,
            }
        except Exception as e:
            return {"mode": "live", "error": str(e)}
    
    def get_open_orders(self) -> list:
        """Get all open orders."""
        if self.mode == ExecutionMode.PAPER:
            return []
        try:
            return self._live_client().get_open_orders()
        except Exception:
            return []
    
    # ---- Order Execution ----
    
    def buy(
        self,
        token_id: str,
        price: float,
        size: float,
    ) -> Optional[str]:
        """
        Submit a BUY order.
        
        Returns: order_id if successful, None on failure.
        """
        if self.mode == ExecutionMode.PAPER:
            return f"paper_buy_{token_id}_{int(time.time())}"
        
        client = self._live_client()
        try:
            from py_clob_client.clob_types import OrderArgs
            response = client.create_and_post_order(
                OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side="BUY",
                ),
            )
            
            if hasattr(response, 'orderID'):
                return response.orderID
            elif isinstance(response, dict):
                return response.get("orderID") or response.get("order_id")
            return str(response)
        except Exception as e:
            print(f"  [EXEC-BUY] Failed: {e}")
            return None
    
    def sell(
        self,
        token_id: str,
        price: float,
        size: float,
    ) -> Optional[str]:
        """
        Submit a SELL order.
        
        Returns: order_id if successful, None on failure.
        """
        if self.mode == ExecutionMode.PAPER:
            return f"paper_sell_{token_id}_{int(time.time())}"
        
        client = self._live_client()
        try:
            from py_clob_client.clob_types import OrderArgs
            response = client.create_and_post_order(
                OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side="SELL",
                ),
            )
            
            if hasattr(response, 'orderID'):
                return response.orderID
            elif isinstance(response, dict):
                return response.get("orderID") or response.get("order_id")
            return str(response)
        except Exception as e:
            print(f"  [EXEC-SELL] Failed: {e}")
            return None
    
    def cancel(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode == ExecutionMode.PAPER:
            return True
        
        try:
            self._live_client().cancel_order(order_id)
            return True
        except Exception as e:
            print(f"  [EXEC-CANCEL] Failed: {e}")
            return False
    
    def cancel_all(self) -> bool:
        """Cancel all open orders."""
        if self.mode == ExecutionMode.PAPER:
            return True
        
        try:
            self._live_client().cancel_all()
            return True
        except Exception as e:
            print(f"  [EXEC-CANCEL-ALL] Failed: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order details."""
        if self.mode == ExecutionMode.PAPER:
            return {"id": order_id, "status": "PAPER"}
        
        try:
            return self._live_client().get_order(order_id)
        except Exception:
            return None
    
    def get_market(self, token_id: str) -> Optional[dict]:
        """Get market info by token ID."""
        try:
            if self.mode == ExecutionMode.LIVE:
                return self._live_client().get_market(token_id)
        except Exception:
            pass
        return None
    
    # ---- Utility ----
    
    def heartbeat(self) -> bool:
        """Check API connection alive."""
        if self.mode == ExecutionMode.PAPER:
            return True
        try:
            self._live_client().post_heartbeat()
            return True
        except Exception:
            return False


# =============================================================================
# FACTORY
# =============================================================================

def create_executor(config: dict) -> Executor:
    """Create an Executor from config dict."""
    mode = config.get("execution_mode", ExecutionMode.PAPER)
    private_key = config.get("private_key") or os.environ.get("POLYMARKET_PRIVATE_KEY")
    funder = config.get("funder_address") or os.environ.get("POLYMARKET_FUNDER")
    fee_slippage = config.get("fee_slippage", 0.0001)
    
    return Executor(
        mode=mode,
        private_key=private_key,
        funder=funder,
        fee_slippage=fee_slippage,
    )
