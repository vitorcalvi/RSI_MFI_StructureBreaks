"""
Optuna + Ray hyper‑parameter optimizer for RSIMFICloudStrategy
============================================================

This script uses **Ray Tune** with **OptunaSearch** to distribute trials across all CPU cores
on Apple‑Silicon (M2). It writes the best parameter set to `best_params.json`.

Prerequisites (tested on macOS 14, Apple M2):
  brew install python  # or ensure you have Python ≥3.10 (universal build)
  python -m pip install --upgrade ray[tune] optuna pandas numpy
  # your back‑tester / data requirements

Run:
  python optimizer.py --trials 500 --output best_params.json  
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

# ---------------------------------------------------------------------------
# 🔧  USER CONFIG SECTION – adjust to your environment
# ---------------------------------------------------------------------------

DATA_PATH = Path("/path/to/your/ohlcv.parquet")  # update!
STARTING_BALANCE = 10_000.0

# ---------------------------------------------------------------------------


def evaluate_strategy(config: Dict[str, Any]) -> Dict[str, float]:
    """Stub – replace with your actual back‑test & PnL calculation.

    Args:
        config: Hyper‑parameters for RSIMFICloudStrategy

    Returns:
        dict with objective value: *Higher* is better.  Use total return, Sharpe, etc.
    """
    # TODO: implement your back‑test – for demo we return random.
    # >>> replace this with something like: run_backtest(prices, config)
    result = float(np.random.normal(loc=0.02, scale=0.01))
    return {"objective": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray Tune + Optuna optimizer for RSIMFICloudStrategy")
    parser.add_argument("--trials", type=int, default=500, help="Number of trials to run")
    parser.add_argument("--output", type=Path, default=Path("best_params.json"), help="Where to store best params")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="rsi_mfi_optimizer", help="Study name")

    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Define search space
    config = {
        # Core lengths
        "rsi_length": tune.choice([8, 10, 12, 14, 16, 18, 20]),
        "mfi_length": tune.randint(5, 16),
        # Thresholds
        "oversold": tune.choice([20, 25, 30, 35]),
        "overbought": tune.choice([65, 70, 75, 80]),
        # Structure
        "structure_lookback": tune.choice([100, 120, 150, 200]),
        "structure_buffer_pct": tune.uniform(0.15, 0.3),
        # Risk / reward
        "fixed_risk_pct": tune.uniform(0.008, 0.015),
        "reward_ratio": tune.uniform(1.8, 2.5),
        "profit_lock_threshold": tune.uniform(1.2, 1.8),
        "trailing_stop_pct": tune.uniform(0.0015, 0.003),
        # Fees / misc (kept constant or narrow band)
        "entry_fee_pct": 0.00025,
        "exit_fee_pct": 0.00025,
        "max_position_time": tune.choice([45, 60, 75, 90, 105, 120]),
        "cooldown_seconds": tune.choice([5, 10, 15, 20, 25, 30]),
        "emergency_stop_pct": tune.uniform(0.005, 0.007),
        "min_profit_distance": tune.uniform(0.001, 0.002),
    }

    # Create OptunaSearch algorithm
    search_alg = OptunaSearch(
        metric="objective",
        mode="max",
        seed=42
    )

    # Run optimization
    tuner = tune.Tuner(
        evaluate_strategy,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=args.trials,
        ),
        param_space=config,
    )

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="objective", mode="max")
    
    print("\nBest value:", best_result.metrics["objective"])
    print("Best params:")
    for k, v in best_result.config.items():
        print(f"  {k}: {v}")

    # Dump to json
    with open(args.output, "w") as f:
        json.dump(best_result.config, f, indent=2)
    print(f"\n✅ Best params saved to {args.output}\n")

    ray.shutdown()


if __name__ == "__main__":
    main()