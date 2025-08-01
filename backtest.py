# eth_hft_backtester_tuner.py (profitable‑export ready)
"""
ETH HFT back‑tester + tuner
===========================

Adds **profitable‑scenario export**:

* `DirectoryTuner.run(save_dir=...)` will copy every evaluated JSON file whose
  `return_pct` > 0 to `save_dir/` and finally zip them to
  `<save_dir>.zip`.
* CLI flag `--jsondir DIR --saveprof OUTDIR` activates this behaviour from the
  command line.

Example
-------
```bash
python3 backtest.py data/eth.csv \
       --jsondir strategies/bot_param_scenarios \
       --saveprof strategies/profitable_scenarios
```
This leaves you with `profitable_scenarios.zip` containing only the profitable
param sets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy back‑tester (same as patched version)
# ---------------------------------------------------------------------------

class EthHFTBacktester:
    MIN_RISK_DIST_PCT = 1e-4

    def __init__(self, data_path: str, params: Dict[str, float], *, initial_balance: float = 10_000.0):
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.params = params.copy()
        self.timeframe = "1min"
        self.data: pd.DataFrame | None = None
        self.trades: List[Dict] = []
        self.balance = initial_balance
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def _load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=["timestamp"], index_col="timestamp")
        df = (
            df.resample(self.timeframe)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self.data = df

    def _compute_indicators(self):
        p = self.params
        close = self.data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / p["rsi_length"], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p["rsi_length"], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.data["rsi"] = 100 - 100 / (1 + rs)
        tp = (self.data["high"] + self.data["low"] + close) / 3
        mf = tp * self.data["volume"]
        sign = tp.diff()
        pos_mf = mf.where(sign > 0, 0)
        neg_mf = mf.where(sign <= 0, 0)
        pos_ema = pos_mf.ewm(alpha=1 / p["mfi_length"], adjust=False).mean()
        neg_ema = neg_mf.ewm(alpha=1 / p["mfi_length"], adjust=False).mean()
        mf_ratio = pos_ema / neg_ema.replace(0, np.nan)
        self.data["mfi"] = 100 - 100 / (1 + mf_ratio)
        self.data[["rsi", "mfi"]] = self.data[["rsi", "mfi"]].fillna(50)

    def run(self):
        self._load_data()
        self._compute_indicators()
        self.trades.clear(); self.balance = self.initial_balance
        self.equity_curve = [(self.data.index[0], self.initial_balance)]
        pos = None; p = self.params
        for ts, row in self.data.iterrows():
            price = row["close"]
            if pos is None:
                candidate = None
                if row["rsi"] < p["oversold"] and row["mfi"] < p["oversold"]:
                    candidate = self._open("BUY", price, ts)
                elif row["rsi"] > p["overbought"] and row["mfi"] > p["overbought"]:
                    candidate = self._open("SELL", price, ts)
                if candidate:
                    pos = candidate
                continue
            if self._manage(pos, price, ts):
                pos = None
        if pos is not None:
            self._close(pos, self.data["close"].iloc[-1], self.data.index[-1], "EOD")
        return {
            "return_pct": (self.balance / self.initial_balance - 1) * 100,
            "max_drawdown_pct": self._max_drawdown_pct(),
        }

    # --- helpers (same logic as earlier)…
    def _open(self, side, price, ts):
        p = self.params
        risk_amt = self.balance * p["fixed_risk_pct"]
        window = self.data.loc[:ts].iloc[-p["structure_lookback"] :]
        stop = window["low"].min() - price * p["structure_buffer_pct"] if side == "BUY" else window["high"].max() + price * p["structure_buffer_pct"]
        risk_dist = abs(price - stop)
        if risk_dist <= price * self.MIN_RISK_DIST_PCT:
            return None
        size = risk_amt / risk_dist
        if not np.isfinite(size) or size <= 0:
            return None
        tp = price + risk_dist * p["reward_ratio"] if side == "BUY" else price - risk_dist * p["reward_ratio"]
        self.balance -= price * size * p["entry_fee_pct"]
        return {"side": side, "entry": price, "stop": stop, "tp": tp, "size": size}

    def _manage(self, pos, price, ts):
        p = self.params
        pnl_pct = ((price - pos["entry"]) / pos["entry"] * 100) if pos["side"] == "BUY" else ((pos["entry"] - price) / pos["entry"] * 100)
        if (pos["side"] == "BUY" and price <= pos["stop"]) or (pos["side"] == "SELL" and price >= pos["stop"]):
            self._close(pos, price, ts, "SL"); return True
        if (pos["side"] == "BUY" and price >= pos["tp"]) or (pos["side"] == "SELL" and price <= pos["tp"]):
            self._close(pos, price, ts, "TP"); return True
        if pnl_pct >= p["profit_lock_threshold"]:
            new_stop = price * (1 - p["trailing_stop_pct"]) if pos["side"] == "BUY" else price * (1 + p["trailing_stop_pct"])
            pos["stop"] = max(pos["stop"], new_stop) if pos["side"] == "BUY" else min(pos["stop"], new_stop)
        return False

    def _close(self, pos, price, ts, reason):
        p = self.params
        pnl = (price - pos["entry"]) * pos["size"] if pos["side"] == "BUY" else (pos["entry"] - price) * pos["size"]
        pnl -= price * pos["size"] * p["exit_fee_pct"]
        self.balance += pnl
        self.trades.append({"pnl": pnl, "close_ts": ts, "reason": reason})
        self.equity_curve.append((ts, self.balance))

    def _max_drawdown_pct(self):
        eq = np.array([x[1] for x in self.equity_curve])
        if eq.size == 0: return 0.0
        peak = np.maximum.accumulate(eq)
        return np.min((eq - peak) / peak) * 100

# ---------------------------------------------------------------------------
# Search utilities (PARAM_SPACE & RandomSearchTuner unchanged)
# ---------------------------------------------------------------------------

PARAM_SPACE: Dict[str, Tuple[float, float, str]] = {
    "rsi_length": (3, 30, "int"),
    "mfi_length": (3, 30, "int"),
    "oversold": (20, 45, "int"),
    "overbought": (55, 80, "int"),
    "structure_lookback": (20, 180, "int"),
    "structure_buffer_pct": (0.0, 0.3, "float"),
    "fixed_risk_pct": (0.005, 0.02, "float"),
    "trailing_stop_pct": (0.0, 0.02, "float"),
    "profit_lock_threshold": (0.5, 4.0, "float"),
    "reward_ratio": (0.5, 10.0, "float"),
    "entry_fee_pct": (0.0001, 0.001, "float"),
    "exit_fee_pct": (0.0001, 0.001, "float"),
}

# RandomSearchTuner identical to previous revision ... (omitted for brevity)

class DirectoryTuner:
    def __init__(self, data_path: str, json_dir: str):
        self.data_path = data_path
        self.json_dir = Path(json_dir)

    def run(self, save_dir: str | None = None):
        best = None
        profitable: List[Tuple[str, Dict]] = []
        for jf in self.json_dir.glob("*.json"):
            params = json.load(open(jf))
            perf = EthHFTBacktester(self.data_path, params).run()
            score = perf["return_pct"] + perf["max_drawdown_pct"]
            print(f"{jf.name:<25}  Ret={perf['return_pct']:>+6.2f}%  DD={perf['max_drawdown_pct']:>+6.2f}%")
            if perf["return_pct"] > 0:
                profitable.append((jf.name, params))
            if best is None or score > best["score"]:
                best = {"file": jf.name, "params": params, **perf, "score": score}
        if save_dir and profitable:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, params in profitable:
                shutil.copy(self.json_dir / name, out_dir / name)
            zip_path = out_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in out_dir.glob("*.json"):
                    zf.write(f, arcname=f.name)
            print(f"\nSaved {len(profitable)} profitable JSONs to {out_dir} and zipped to {zip_path}")
        return best

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "
