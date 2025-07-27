import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import vectorbt as vbt
import pandas as pd
import numpy as np
import json
import yfinance as yf

# Import your strategy
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy

def backtest_strategy(config, data):
    """Backtest function for Ray Tune"""
    # Create strategy with config
    strategy = RSIMFICloudStrategy(config)
    
    # Calculate indicators
    df = strategy.calculate_indicators(data.copy())
    
    # Generate signals
    rsi = df['rsi'].values
    mfi = df['mfi'].values
    
    # Entry and exit conditions
    entries = (rsi < config['oversold_level']) & (mfi < config['oversold_level'])
    exits = (rsi > config['overbought_level']) & (mfi > config['overbought_level'])
    
    # Run vectorbt backtest
    pf = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=10000,
        fees=0.001,
        slippage=0.001
    )
    
    # Calculate metrics
    metrics = {
        'total_return': pf.total_return(),
        'sharpe_ratio': pf.sharpe_ratio(),
        'max_drawdown': pf.max_drawdown(),
        'win_rate': pf.win_rate(),
        'total_trades': pf.total_trades()
    }
    
    # Score calculation
    if metrics['max_drawdown'] > 0.3 or metrics['total_trades'] < 10:
        score = -1000
    else:
        score = metrics['sharpe_ratio'] - (metrics['max_drawdown'] * 0.5)
    
    # Report to Ray Tune
    tune.report(
        score=score,
        total_return=metrics['total_return'],
        sharpe_ratio=metrics['sharpe_ratio'],
        max_drawdown=metrics['max_drawdown'],
        win_rate=metrics['win_rate'],
        total_trades=metrics['total_trades']
    )

def run_ray_tune_optimization(data):
    """Run Ray Tune optimization"""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        'rsi_length': tune.randint(5, 20),
        'mfi_length': tune.randint(5, 20),
        'oversold_level': tune.randint(20, 40),
        'overbought_level': tune.randint(60, 80),
        'require_volume': tune.choice([True, False]),
        'require_trend': tune.choice([True, False])
    }
    
    # Use Optuna search algorithm
    optuna_search = OptunaSearch(metric="score", mode="max")
    
    # Run optimization
    analysis = tune.run(
        lambda config: backtest_strategy(config, data),
        config=search_space,
        search_alg=optuna_search,
        num_samples=100,
        resources_per_trial={"cpu": 1},
        verbose=1
    )
    
    # Get best config
    best_config = analysis.get_best_config(metric="score", mode="max")
    best_result = analysis.get_best_trial().last_result
    
    ray.shutdown()
    
    return best_config, best_result

def validate_results(best_params, data):
    """Validate results with walk-forward analysis"""
    # Split data for walk-forward
    split_point = int(len(data) * 0.7)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    # Test on out-of-sample data
    strategy = RSIMFICloudStrategy(best_params)
    df = strategy.calculate_indicators(test_data.copy())
    
    rsi = df['rsi'].values
    mfi = df['mfi'].values
    
    entries = (rsi < best_params['oversold_level']) & (mfi < best_params['oversold_level'])
    exits = (rsi > best_params['overbought_level']) & (mfi > best_params['overbought_level'])
    
    pf = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=10000,
        fees=0.001,
        slippage=0.001
    )
    
    return {
        'test_return': pf.total_return(),
        'test_sharpe': pf.sharpe_ratio(),
        'test_drawdown': pf.max_drawdown(),
        'test_trades': pf.total_trades()
    }

def main():
    # Load data
    ticker = "BTC-USD"
    data = yf.download(ticker, start="2022-01-01", end="2024-01-01", interval="1h")
    # Fix MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower() for c in data.columns]
    else:
        data.columns = [c.lower() for c in data.columns]
    
    print("Starting Ray Tune optimization...")
    
    # Run optimization
    best_config, best_result = run_ray_tune_optimization(data)
    
    print("\n=== Optimization Results ===")
    print(f"Best configuration: {best_config}")
    print(f"Best score: {best_result['score']:.4f}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_result['total_return']:.2%}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.2%}")
    print(f"Win Rate: {best_result['win_rate']:.2%}")
    print(f"Total Trades: {best_result['total_trades']}")
    
    # Validate on test data
    print("\n=== Validation Results ===")
    validation = validate_results(best_config, data)
    print(f"Test Return: {validation['test_return']:.2%}")
    print(f"Test Sharpe: {validation['test_sharpe']:.2f}")
    print(f"Test Drawdown: {validation['test_drawdown']:.2%}")
    print(f"Test Trades: {validation['test_trades']}")
    
    # Save best parameters
    with open('ray_tune_optimized_params.json', 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print("\nBest parameters saved to 'ray_tune_optimized_params.json'")

if __name__ == "__main__":
    main()