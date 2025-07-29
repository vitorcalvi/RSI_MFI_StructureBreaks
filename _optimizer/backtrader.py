#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import pandas as pd
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
except ImportError:
    print("Error: Cannot import RSIMFICloudStrategy")
    sys.exit(1)

class HFTOptimizer:
    def __init__(self):
        self.client = None  # Use synthetic data
        
    def generate_realistic_data(self, periods=1500):
        """Generate realistic ZORA-like 5m data"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-07-01', periods=periods, freq='5min')
        
        # More realistic ZORA price action
        base_price = 0.085
        
        # Add multiple market regimes
        trend_changes = np.random.choice([0, 1], size=periods//200, p=[0.7, 0.3])
        regime_returns = []
        
        for i, change in enumerate(trend_changes):
            length = min(200, periods - i*200)
            if change == 1:  # Trending
                trend_strength = np.random.uniform(-0.0005, 0.0005)
                regime_ret = np.random.normal(trend_strength, 0.0025, length)
            else:  # Ranging
                regime_ret = np.random.normal(0, 0.0015, length)
            regime_returns.extend(regime_ret)
        
        # Pad if needed
        while len(regime_returns) < periods:
            regime_returns.append(np.random.normal(0, 0.002))
        
        regime_returns = np.array(regime_returns[:periods])
        
        # Add volatility clustering
        vol_base = 0.002
        vol_persistence = 0.9
        volatility = [vol_base]
        
        for i in range(1, periods):
            vol_shock = np.random.normal(0, 0.0005)
            new_vol = vol_persistence * volatility[-1] + (1-vol_persistence) * vol_base + vol_shock
            volatility.append(max(0.0005, new_vol))
        
        volatility = np.array(volatility)
        
        # Generate price path
        price_returns = regime_returns * volatility
        prices = base_price * np.exp(np.cumsum(price_returns))
        
        # Generate OHLC with realistic spreads
        opens = prices
        spreads = np.random.uniform(0.0005, 0.002, periods)
        
        highs = opens * (1 + spreads + np.abs(np.random.normal(0, volatility)))
        lows = opens * (1 - spreads - np.abs(np.random.normal(0, volatility)))
        closes = opens + np.random.normal(0, volatility/2) * opens
        
        # Ensure OHLC consistency
        for i in range(periods):
            high_val = max(opens[i], closes[i], highs[i])
            low_val = min(opens[i], closes[i], lows[i])
            highs[i] = high_val
            lows[i] = low_val
        
        volumes = np.random.lognormal(12, 0.8, periods)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows, 
            'close': closes,
            'volume': volumes
        }, index=dates)

class OptimizedBacktester:
    def __init__(self):
        pass
        
    def backtest_strategy(self, df, params):
        """Realistic backtest with proper HFT logic"""
        strategy = RSIMFICloudStrategy()
        
        # Update strategy params
        for key, value in params.items():
            strategy.params[key] = value
        
        strategy.atr_multiplier = params.get('atr_multiplier', 1.2)
        strategy.signal_cooldown_period = params.get('signal_cooldown', 2)
        
        # Calculate indicators
        df = strategy.calculate_indicators(df)
        
        if len(df) < 100:
            return self._empty_metrics()
        
        # Trading simulation
        balance = 1000
        position = 0
        trades = []
        entry_price = 0
        entry_time = None
        last_signal_bar = -999
        
        fee_rate = 0.001  # 0.1% per trade (realistic for spot)
        
        for i in range(50, len(df)):  # Skip first 50 bars
            row = df.iloc[i]
            current_time = df.index[i]
            
            # Skip if in cooldown
            if i - last_signal_bar < params.get('signal_cooldown', 2):
                continue
            
            rsi = row['rsi']
            mfi = row['mfi']
            trend = row['trend']
            price = row['close']
            
            if pd.isna(rsi) or pd.isna(mfi):
                continue
            
            # Entry signals
            if position == 0:
                # Long entry
                if (rsi < params['oversold_level'] and 
                    mfi < params['oversold_level']):
                    
                    if not params.get('require_trend', True) or trend == 'UP':
                        position = 1
                        entry_price = price * (1 + fee_rate)  # Include slippage
                        entry_time = current_time
                        last_signal_bar = i
                
                # Short entry  
                elif (rsi > params['overbought_level'] and
                      mfi > params['overbought_level']):
                    
                    if not params.get('require_trend', True) or trend == 'DOWN':
                        position = -1
                        entry_price = price * (1 - fee_rate)  # Include slippage
                        entry_time = current_time
                        last_signal_bar = i
            
            # Exit signals
            elif position != 0:
                exit_signal = False
                exit_reason = ""
                
                # Opposite signal exit
                if position == 1 and rsi > params['overbought_level']:
                    exit_signal = True
                    exit_reason = "RSI_OVERBOUGHT"
                elif position == -1 and rsi < params['oversold_level']:
                    exit_signal = True
                    exit_reason = "RSI_OVERSOLD"
                
                # Time-based exit (max 2 hours for HFT)
                elif (current_time - entry_time).total_seconds() > 7200:
                    exit_signal = True
                    exit_reason = "TIME_LIMIT"
                
                # ATR-based stop loss
                atr = row['atr']
                if not pd.isna(atr) and atr > 0:
                    stop_distance = atr * params.get('atr_multiplier', 1.2)
                    
                    if position == 1 and price <= entry_price - stop_distance:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                    elif position == -1 and price >= entry_price + stop_distance:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                
                if exit_signal:
                    exit_price = price * (1 - fee_rate * position)  # Include fees & slippage
                    
                    # Calculate PnL
                    if position == 1:  # Long
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:  # Short
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    duration_minutes = (current_time - entry_time).total_seconds() / 60
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl_pct': pnl_pct,
                        'duration_minutes': duration_minutes,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    last_signal_bar = i
        
        return self._calculate_performance(trades)
    
    def _empty_metrics(self):
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'num_trades': 0,
            'avg_duration': 0,
            'calmar_ratio': 0
        }
    
    def _calculate_performance(self, trades):
        if not trades:
            return self._empty_metrics()
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_return = df['pnl_pct'].sum()
        num_trades = len(df)
        win_rate = (df['pnl_pct'] > 0).mean()
        avg_duration = df['duration_minutes'].mean()
        
        # Profit factor
        wins = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
        losses = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = wins / losses if losses > 0 else 2.0
        
        # Risk metrics
        returns = df['pnl_pct']
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 288)  # 5min periods
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else total_return
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'avg_duration': avg_duration,
            'calmar_ratio': calmar
        }

def grid_search_optimization():
    """Optimized grid search with proper HFT bounds"""
    print("Starting HFT Grid Search Optimization...")
    
    optimizer = HFTOptimizer()
    backtester = OptimizedBacktester()
    
    # Generate realistic data
    print("Generating realistic 5m ZORA data...")
    data = optimizer.generate_realistic_data(1500)
    print(f"Data shape: {data.shape}")
    
    # Proper HFT parameter grid based on research
    param_grid = {
        'rsi_length': [5, 7, 9],           # Short periods for HFT
        'mfi_length': [5, 7, 9],           # Short periods for HFT  
        'oversold_level': [20, 25, 30],    # More reasonable levels
        'overbought_level': [70, 75, 80],  # More reasonable levels
        'atr_multiplier': [1.0, 1.2, 1.5], # Tighter stops for HFT
        'signal_cooldown': [1, 2, 3],      # Short cooldown
        'require_trend': [True, False]
    }
    
    best_score = -999
    best_params = None
    all_results = []
    
    total_combinations = (len(param_grid['rsi_length']) * 
                         len(param_grid['mfi_length']) * 
                         len(param_grid['oversold_level']) * 
                         len(param_grid['overbought_level']) * 
                         len(param_grid['atr_multiplier']) * 
                         len(param_grid['signal_cooldown']) * 
                         len(param_grid['require_trend']))
    
    print(f"Testing {total_combinations} parameter combinations...")
    count = 0
    
    for rsi_len in param_grid['rsi_length']:
        for mfi_len in param_grid['mfi_length']:
            for oversold in param_grid['oversold_level']:
                for overbought in param_grid['overbought_level']:
                    if overbought <= oversold + 30:  # Ensure reasonable spread
                        continue
                        
                    for atr_mult in param_grid['atr_multiplier']:
                        for cooldown in param_grid['signal_cooldown']:
                            for trend_req in param_grid['require_trend']:
                                count += 1
                                
                                params = {
                                    'rsi_length': rsi_len,
                                    'mfi_length': mfi_len,
                                    'oversold_level': oversold,
                                    'overbought_level': overbought,
                                    'atr_multiplier': atr_mult,
                                    'signal_cooldown': cooldown,
                                    'require_trend': trend_req
                                }
                                
                                try:
                                    # Walk-forward test
                                    train_data = data.iloc[:1000]  # First 1000 bars for training
                                    test_data = data.iloc[1000:1300]  # Next 300 for testing
                                    
                                    # Quick train validation
                                    train_metrics = backtester.backtest_strategy(train_data, params)
                                    
                                    if train_metrics['num_trades'] < 5:
                                        continue
                                    
                                    # Test on unseen data
                                    test_metrics = backtester.backtest_strategy(test_data, params)
                                    
                                    # Combined score favoring consistency
                                    score = (
                                        test_metrics['total_return'] * 0.25 +
                                        test_metrics['sharpe_ratio'] * 0.25 +
                                        test_metrics['win_rate'] * 0.20 +
                                        test_metrics['profit_factor'] * 0.15 +
                                        min(test_metrics['calmar_ratio'], 2) * 0.15
                                    )
                                    
                                    # Penalty for too many trades (overtrading)
                                    if test_metrics['num_trades'] > 100:
                                        score *= 0.8
                                    
                                    all_results.append({
                                        'params': params.copy(),
                                        'train_metrics': train_metrics,
                                        'test_metrics': test_metrics,
                                        'score': score
                                    })
                                    
                                    if (score > best_score and 
                                        test_metrics['num_trades'] >= 5 and
                                        test_metrics['win_rate'] > 0.4):
                                        best_score = score
                                        best_params = params.copy()
                                    
                                    if count % 20 == 0:
                                        print(f"Progress: {count}/{total_combinations} - Best score: {best_score:.4f}")
                                        
                                except Exception as e:
                                    continue
    
    # Sort results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*60)
    print("HFT OPTIMIZATION COMPLETE")
    print("="*60)
    
    if best_params:
        print(f"Best Score: {best_score:.4f}")
        print(f"Total valid combinations tested: {len(all_results)}")
        
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Show best performance
        best_result = all_results[0]
        test_perf = best_result['test_metrics']
        
        print(f"\nOut-of-Sample Performance:")
        print(f"  Total Return: {test_perf['total_return']:.2%}")
        print(f"  Sharpe Ratio: {test_perf['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {test_perf['win_rate']:.2%}")
        print(f"  Max Drawdown: {test_perf['max_drawdown']:.2%}")
        print(f"  Profit Factor: {test_perf['profit_factor']:.2f}")
        print(f"  Number of Trades: {test_perf['num_trades']}")
        print(f"  Avg Duration: {test_perf['avg_duration']:.1f} minutes")
        
        # Save results
        with open('optimized_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save top 5 for ensemble
        top_5_params = [r['params'] for r in all_results[:5]]
        with open('ensemble_params.json', 'w') as f:
            json.dump(top_5_params, f, indent=2)
        
        print(f"\nFiles saved:")
        print(f"  - optimized_params.json (best parameters)")
        print(f"  - ensemble_params.json (top 5 parameter sets)")
        
        return best_params, all_results
    else:
        print("No valid parameters found!")
        return None, all_results

if __name__ == "__main__":
    best_params, results = grid_search_optimization()