import ray
from ray import tune

import vectorbt as vbt
import pandas as pd
import numpy as np
import json
import ccxt
from datetime import datetime, timedelta

# Import your strategy
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy

def fetch_bybit_data(symbol='SOL/USDT', timeframe='h', days=365):
    """Fetch data from Bybit"""
    # Initialize Bybit exchange without credentials for public data
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })
    
    try:
        # Calculate since timestamp
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        return df
    except Exception as e:
        print(f"Error fetching data from Bybit: {e}")
        raise

def backtest_strategy(config, data):
    """Backtest function for Ray Tune"""
    try:
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
        total_return = pf.total_return()
        sharpe_ratio = pf.sharpe_ratio() if pf.sharpe_ratio() is not None else 0
        max_drawdown = pf.max_drawdown()
        total_trades = len(pf.trades.records)  # Use len of trades records instead of count()
        
        # Calculate win rate manually
        trades = pf.trades.records
        if len(trades) > 0:
            winning_trades = len(trades[trades['pnl'] > 0])
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        # Score calculation
        if pd.isna(sharpe_ratio) or pd.isna(max_drawdown):
            score = -1000
        elif max_drawdown > 0.3 or total_trades < 10:
            score = -1000
        else:
            score = sharpe_ratio - (max_drawdown * 0.5)
        
        # Report to Ray Tune - use train.report with metrics dict
        tune.report({
            'score': score,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        })
    except Exception as e:
        print(f"Error in backtest: {e}")
        # Report failed run
        tune.report({
            'score': -9999,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 1,
            'win_rate': 0,
            'total_trades': 0
        })

def run_ray_tune_optimization(data):
    """Run Ray Tune optimization"""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
    'rsi_length': tune.randint(10, 30),      # Wider range
    'mfi_length': tune.randint(10, 30),      
    'oversold_level': tune.randint(15, 35),  
    'overbought_level': tune.randint(65, 85),
    'require_volume': tune.choice([True, False]),
    'require_trend': tune.choice([True, False])
}
    
    # Run optimization (num_samples=5 for testing, use 100+ for production)
    analysis = tune.run(
        lambda config: backtest_strategy(config, data),
        config=search_space,
        num_samples=100,  # Reduced for testing, use 100 for production
        resources_per_trial={"cpu": 5},
        verbose=1
    )
    
    # Get best config
    try:
        best_config = analysis.get_best_config(metric="score", mode="max")
        best_trial = analysis.get_best_trial(metric="score", mode="max")
        if best_trial:
            best_result = best_trial.last_result
        else:
            print("No successful trials found!")
            ray.shutdown()
            return None, None
    except Exception as e:
        print(f"Error getting best config: {e}")
        ray.shutdown()
        return None, None
    
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
    
    # Drop NaN values from indicators
    df = df.dropna(subset=['rsi', 'mfi'])
    
    if len(df) < 10:
        print(f"Warning: Not enough test data after indicator calculation: {len(df)} rows")
        return {
            'test_return': 0,
            'test_sharpe': 0,
            'test_drawdown': 0,
            'test_trades': 0
        }
    
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
    
    # Calculate metrics
    total_return = pf.total_return()
    sharpe_ratio = pf.sharpe_ratio() if pf.sharpe_ratio() is not None else 0
    max_drawdown = pf.max_drawdown()
    total_trades = len(pf.trades.records)  # Use len of trades records
    
    return {
        'test_return': total_return,
        'test_sharpe': sharpe_ratio,
        'test_drawdown': max_drawdown,
        'test_trades': total_trades
    }

def main():
    # Load data from Bybit
    symbol = "SOL/USDT"
    print(f"Fetching {symbol} data from Bybit...")
    
    try:
        data = fetch_bybit_data(symbol=symbol, timeframe='1h', days=365)
        print(f"Data fetched: {len(data)} candles")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Test single backtest before optimization
    print("\nTesting single backtest...")
    test_config = {
        'rsi_length': 14,
        'mfi_length': 14,
        'oversold_level': 30,
        'overbought_level': 70,
        'require_volume': True,
        'require_trend': False
    }
    
    try:
        # Create a simple test without Ray
        strategy = RSIMFICloudStrategy(test_config)
        df = strategy.calculate_indicators(data.copy())
        print(f"Indicators calculated successfully")
        print(f"RSI range: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
        print(f"MFI range: {df['mfi'].min():.2f} - {df['mfi'].max():.2f}")
    except Exception as e:
        print(f"Error in test backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStarting Ray Tune optimization...")
    
    # Run optimization
    best_config, best_result = run_ray_tune_optimization(data)
    
    if best_config is None or best_result is None:
        print("Optimization failed - no successful trials")
        return
    
    print("\n=== Optimization Results ===")
    print(f"Best configuration: {best_config}")
    print(f"Best score: {best_result['score']:.4f}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_result['total_return']:.2%}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.2%}")
    if 'win_rate' in best_result:
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