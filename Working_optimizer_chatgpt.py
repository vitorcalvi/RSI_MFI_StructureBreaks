"""
Diagnostic RSI-MFI Optimizer - Full fixed version
"""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

# USER CONFIG
STARTING_BALANCE = 10_000.0


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14
) -> pd.Series:
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = (
        money_flow.where(typical_price > typical_price.shift(1), 0)
        .rolling(window=length)
        .sum()
    )
    negative_flow = (
        money_flow.where(typical_price < typical_price.shift(1), 0)
        .rolling(window=length)
        .sum()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    mfi = mfi.fillna(50)  # neutral fallback
    return mfi


def find_structure_levels(
    high: pd.Series, low: pd.Series, lookback: int, buffer_pct: float
) -> Tuple[pd.Series, pd.Series]:
    """Find support and resistance levels"""
    resistance = high.rolling(window=lookback).max() * (1 + buffer_pct)
    support = low.rolling(window=lookback).min() * (1 - buffer_pct)
    return support, resistance


def backtest_strategy(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """Backtest RSI-MFI strategy with given parameters"""
    if len(data) < max(
        config["rsi_length"], config["mfi_length"], config["structure_lookback"]
    ) + 50:
        return {
            "objective": -1.0,
            "total_return": -1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
        }

    # Indicators
    data["rsi"] = calculate_rsi(data["close"], config["rsi_length"])
    data["mfi"] = calculate_mfi(
        data["high"], data["low"], data["close"], data["volume"], config["mfi_length"]
    )
    data["support"], data["resistance"] = find_structure_levels(
        data["high"], data["low"], config["structure_lookback"], config["structure_buffer_pct"]
    )

    balance = STARTING_BALANCE
    position = 0.0  # number of units
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trailing_stop = 0.0
    entry_time = 0
    last_exit_time = 0

    entry_cost = 0.0  # amount invested (principal)
    entry_fee_paid = 0.0  # entry fee
    trades = []
    balances = [balance]

    # Debug counters
    rsi_signals = 0
    mfi_signals = 0
    support_signals = 0
    combined_signals = 0

    for i in range(len(data)):
        if i < config["structure_lookback"]:
            balances.append(balance)
            continue

        row = data.iloc[i]
        current_price = row["close"]
        current_time = i

        # EXIT logic
        if position != 0:
            exit_signal = False
            exit_reason = ""

            if current_time - entry_time >= config["max_position_time"]:
                exit_signal = True
                exit_reason = "time_limit"
            elif current_price <= stop_loss:
                exit_signal = True
                exit_reason = "stop_loss"
            elif current_price >= take_profit:
                exit_signal = True
                exit_reason = "take_profit"
            elif current_price <= trailing_stop:
                exit_signal = True
                exit_reason = "trailing_stop"
            elif (
                entry_price > 0
                and (entry_price - current_price) / entry_price >= config["emergency_stop_pct"]
            ):
                exit_signal = True
                exit_reason = "emergency_stop"

            if exit_signal:
                exit_value = position * current_price
                exit_fee = exit_value * config["exit_fee_pct"]

                pnl = exit_value - entry_cost - entry_fee_paid - exit_fee

                # Return principal + profit minus exit fee
                balance += exit_value - exit_fee

                trades.append(
                    {
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl,
                        "exit_reason": exit_reason,
                    }
                )

                # reset position state
                position = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                entry_fee_paid = 0.0
                trailing_stop = 0.0
                last_exit_time = current_time
            else:
                if current_price > entry_price * config["profit_lock_threshold"]:
                    trailing_stop = max(
                        trailing_stop, current_price * (1 - config["trailing_stop_pct"])
                    )

        # ENTRY logic (lenient)
        if position == 0 and current_time - last_exit_time >= config["cooldown_seconds"]:
            rsi_signal = row["rsi"] <= config["oversold"]
            mfi_signal = row["mfi"] <= config["oversold"]
            support_signal = current_price <= row["support"] * 1.5  # very lenient

            if rsi_signal:
                rsi_signals += 1
            if mfi_signal:
                mfi_signals += 1
            if support_signal:
                support_signals += 1

            if rsi_signal or mfi_signal:
                combined_signals += 1

                # Desired stop and profit
                stop_distance_pct = config["fixed_risk_pct"]
                stop_loss_candidate = current_price * (1 - stop_distance_pct)
                take_profit_candidate = current_price * (1 + stop_distance_pct * config["reward_ratio"])

                # Position sizing with upfront fee accounted: ensure investment + entry_fee <= balance
                # Let investment = max_investment = balance / (1 + entry_fee_pct)
                max_investment = balance / (1 + config["entry_fee_pct"])
                position_size = max_investment / current_price  # number of units
                entry_fee = max_investment * config["entry_fee_pct"]
                entry_cost_candidate = position_size * current_price  # equals max_investment

                # Very lenient minimum profit distance
                if (take_profit_candidate - current_price) / current_price >= config.get(
                    "min_profit_distance", 0.001
                ):
                    # Only enter if we have enough to pay investment + fee (should be guaranteed by formula)
                    if balance >= entry_cost_candidate + entry_fee - 1e-12:
                        # Commit
                        position = position_size
                        entry_price = current_price
                        entry_time = current_time
                        stop_loss = stop_loss_candidate
                        take_profit = take_profit_candidate
                        trailing_stop = stop_loss_candidate
                        entry_cost = entry_cost_candidate
                        entry_fee_paid = entry_fee

                        # Deduct investment and entry fee from cash balance
                        balance -= entry_cost + entry_fee

        # Portfolio value: cash + current position value
        portfolio_value = balance + (position * current_price if position > 0 else 0)
        balances.append(portfolio_value)

    # Final exit if still in position
    if position != 0:
        final_price = data.iloc[-1]["close"]
        exit_value = position * final_price
        exit_fee = exit_value * config["exit_fee_pct"]

        pnl = exit_value - entry_cost - entry_fee_paid - exit_fee
        balance += exit_value - exit_fee

        trades.append(
            {
                "entry_price": entry_price,
                "exit_price": final_price,
                "pnl": pnl,
                "exit_reason": "final_exit",
            }
        )

        position = 0.0
        entry_price = 0.0
        entry_cost = 0.0
        entry_fee_paid = 0.0
        trailing_stop = 0.0

    balances = np.array(balances)
    if len(balances) < 2:
        return {
            "objective": -1.0,
            "total_return": -1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
        }

    total_return = (balances[-1] - STARTING_BALANCE) / STARTING_BALANCE

    returns = np.diff(balances) / balances[:-1]
    returns = returns[~np.isnan(returns)]

    sharpe_ratio = (
        np.mean(returns) / np.std(returns) * np.sqrt(252)
        if len(returns) > 0 and np.std(returns) > 0
        else 0.0
    )

    running_max = np.maximum.accumulate(balances)
    drawdowns = (balances - running_max) / running_max
    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    winning_trades = len([t for t in trades if t["pnl"] > 0])
    total_trades = len(trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    if total_trades < 1 or total_return < -0.8:
        objective = -1.0
    else:
        objective = (
            total_return * 0.5
            + sharpe_ratio * 0.2
            + win_rate * 0.2
            - max_drawdown * 0.1
        )

    return {
        "objective": float(objective),
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "total_trades": total_trades,
        # Debug info
        "rsi_signals": rsi_signals,
        "mfi_signals": mfi_signals,
        "support_signals": support_signals,
        "combined_signals": combined_signals,
        "min_rsi": float(data["rsi"].min()) if not data["rsi"].isna().all() else 100,
        "min_mfi": float(data["mfi"].min()) if not data["mfi"].isna().all() else 100,
    }


def load_data(data_path: Union[Path, str]) -> pd.DataFrame:
    """Load OHLCV data"""
    data_path = Path(data_path)
    if data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")

    if "timestamp" in data.columns:
        data = data.sort_values("timestamp")

    return data.reset_index(drop=True)


def evaluate_strategy(
    config: Dict[str, Any], data_path: Union[Path, str] = None
) -> Dict[str, float]:
    """Load data and run backtest"""
    if data_path is None:
        data_path = Path("./_data/ADAUSDT_1_7d_20250731_071705.csv")
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"[error] could not find data file at {data_path!r} inside evaluate_strategy")
        return {
            "objective": -1.0,
            "total_return": -1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
        }
    except Exception as e:
        print(f"[error] failed to load data from {data_path!r}: {e}")
        return {
            "objective": -1.0,
            "total_return": -1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
        }

    return backtest_strategy(data, config)


def evaluate_with_path(config):
    # Worker-level normalization and debug logging
    data_path_raw = config.get("data_path")
    print(f"[worker debug] cwd: {Path.cwd()}")
    print(f"[worker debug] raw data_path from config: {data_path_raw!r}")
    data_path = Path(data_path_raw).expanduser().resolve()
    print(f"[worker debug] resolved data_path: {data_path}, exists: {data_path.is_file()}")
    return evaluate_strategy(config, data_path)


def signal_handler(signum, frame):
    """Gracefully handle interrupts"""
    print("\nReceived interrupt signal. Shutting down...")
    if ray.is_initialized():
        ray.shutdown()
    exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Diagnostic RSI-MFI optimizer")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument(
        "--output", type=Path, default=Path("best_params.json"), help="Where to store best params"
    )
    parser.add_argument("--data", type=Path, help="Path to data file (CSV or parquet)")

    args = parser.parse_args()

    # Resolve and validate data path early
    data_path = Path(args.data if args.data else "./_data/ADAUSDT_1_7d_20250731_071705.csv").expanduser().resolve()
    if not data_path.is_file():
        print(f"Error: data file '{data_path}' does not exist. Aborting.")
        return

    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        config = {
            "rsi_length": tune.choice([14]),
            "mfi_length": tune.choice([14]),
            "oversold": tune.choice([35, 40, 45, 50]),
            "overbought": tune.choice([60, 65, 70]),
            "structure_lookback": tune.choice([50]),
            "structure_buffer_pct": tune.choice([0.1]),
            "fixed_risk_pct": tune.choice([0.02]),
            "reward_ratio": tune.choice([2.0]),
            "profit_lock_threshold": tune.choice([1.2]),
            "trailing_stop_pct": tune.choice([0.005]),
            "entry_fee_pct": 0.0055,
            "exit_fee_pct": 0.0055,
            "max_position_time": tune.choice([60]),
            "cooldown_seconds": tune.choice([5]),
            "emergency_stop_pct": tune.choice([0.01]),
            "min_profit_distance": tune.choice([0.001]),
            "data_path": str(data_path),
        }

        search_alg = OptunaSearch(metric="objective", mode="max", seed=42)

        tuner = tune.Tuner(
            evaluate_with_path,
            tune_config=tune.TuneConfig(search_alg=search_alg, num_samples=args.trials),
            param_space=config,
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="objective", mode="max")

        print("\n" + "=" * 60)
        print("DIAGNOSTIC RESULTS")
        print("=" * 60)
        print(f"Best objective: {best_result.metrics.get('objective', 'N/A')}")
        print(f"Total trades: {best_result.metrics.get('total_trades', 0)}")
        print(f"Total return: {best_result.metrics.get('total_return', 0):.4f}")

        print(f"\nDEBUG INFO:")
        print(f"RSI signals: {best_result.metrics.get('rsi_signals', 0)}")
        print(f"MFI signals: {best_result.metrics.get('mfi_signals', 0)}")
        print(f"Combined signals: {best_result.metrics.get('combined_signals', 0)}")
        print(f"Min RSI in data: {best_result.metrics.get('min_rsi', 'N/A')}")
        print(f"Min MFI in data: {best_result.metrics.get('min_mfi', 'N/A')}")

        print(f"\nBest params:")
        output_config = {k: v for k, v in best_result.config.items() if k != "data_path"}
        for k, v in output_config.items():
            print(f"  {k}: {v}")

        try:
            with open(args.output, "w") as f:
                json.dump(output_config, f, indent=2)
            print(f"\n✅ Best params saved to {args.output}")
        except Exception as e:
            print(f"\nFailed to save best params to {args.output}: {e}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"Error during optimization: {e}")
    finally:
        try:
            if ray.is_initialized():
                ray.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()
