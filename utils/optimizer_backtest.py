import os
import sys
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy

class BayesianBacktestOptimizer:
    def __init__(self, h5_path='data/crypto_database.h5', symbol='SOLUSDT', timeframe='1_30d'):
        self.h5_path = h5_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_class = RSIMFICloudStrategy
        
        self.initial_balance = 10000
        self.commission = 0.001
        self.slippage = 0.0005
        
        self.data = self._load_data()
        
        # Reduce data for faster optimization
        if len(self.data) > 10000:
            print(f"Reducing data from {len(self.data)} to last 10000 candles for optimization")
            self.data = self.data.iloc[-10000:]
        
        self.train_size = 0.70
        self.val_size = 0.15
        self.test_size = 0.15
        self._split_data()
        
    def _load_data(self):
        print(f"Loading data from {self.h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            dataset_path = f'{self.symbol}/{self.timeframe}'
            table_path = f'{dataset_path}/table'
            table = f[table_path][:]
            
            timestamps = table['values_block_0'][:, 0] / 1e9
            ohlcv = table['values_block_1']
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': ohlcv[:, 0],
                'high': ohlcv[:, 1], 
                'low': ohlcv[:, 2],
                'close': ohlcv[:, 3],
                'volume': ohlcv[:, 4],
                'volume_usdt': ohlcv[:, 5]
            })
            
            df.set_index('timestamp', inplace=True)
            df = df.dropna()
            print(f"Loaded {len(df)} candles")
            return df
    
    def _split_data(self):
        n = len(self.data)
        train_end = int(n * self.train_size)
        val_end = int(n * (self.train_size + self.val_size))
        
        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]
    
    def backtest(self, params, data):
        print(f"Starting backtest with {len(data)} candles...")
        strategy = self.strategy_class(params)
        
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []
        
        # Start from minimum required candles for indicators
        start_idx = max(50, params.get('rsi_length', 14) + params.get('mfi_length', 14))
        
        for i in range(start_idx, len(data)):
            if i % 1000 == 0:  # Progress indicator
                print(f"Progress: {i}/{len(data)} candles processed")
            
            # Only pass last 100 candles to strategy for efficiency
            lookback = min(i+1, 100)
            current_data = data.iloc[i+1-lookback:i+1]
            signal = strategy.generate_signal(current_data)
            current_data = data.iloc[:i+1]
            signal = strategy.generate_signal(current_data)
            
            if signal:
                current_price = data['close'].iloc[i]
                current_time = data.index[i]
                
                if signal['action'] == 'BUY' and position is None:
                    entry_price = current_price * (1 + self.slippage)
                    position_size = (balance * 0.95) / entry_price
                    commission_paid = position_size * entry_price * self.commission
                    
                    position = {
                        'entry_price': entry_price,
                        'size': position_size,
                        'entry_time': current_time
                    }
                    
                    balance -= (position_size * entry_price + commission_paid)
                    
                elif signal['action'] == 'SELL' and position is not None:
                    exit_price = current_price * (1 - self.slippage)
                    position_value = position['size'] * exit_price
                    commission_paid = position_value * self.commission
                    
                    pnl = position_value - (position['size'] * position['entry_price'])
                    balance += (position_value - commission_paid)
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position['size'] * position['entry_price'])) * 100
                    })
                    
                    position = None
            
            current_equity = balance
            if position:
                current_equity += position['size'] * data['close'].iloc[i]
            equity_curve.append(current_equity)
        
        if position:
            exit_price = data['close'].iloc[-1] * (1 - self.slippage)
            position_value = position['size'] * exit_price
            commission_paid = position_value * self.commission
            pnl = position_value - (position['size'] * position['entry_price'])
            balance += (position_value - commission_paid)
            
            trades.append({
                'pnl': pnl,
                'pnl_pct': (pnl / (position['size'] * position['entry_price'])) * 100
            })
        
        # Add initial balance at the beginning
        equity_curve = [self.initial_balance] + equity_curve
        
        metrics = self._calculate_metrics(trades, equity_curve)
        return metrics, trades
    
    def _calculate_metrics(self, trades, equity_curve):
        if len(trades) == 0:
            return {
                'sharpe_ratio': -999,
                'total_return': -1,
                'win_rate': 0,
                'num_trades': 0
            }
        
        equity_curve = np.array(equity_curve)
        
        # Calculate returns safely
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] > 0:
                    ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                    returns.append(ret)
            returns = np.array(returns)
        else:
            returns = np.array([])
        
        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance
        
        if len(returns) > 1 and np.std(returns) > 1e-8:
            periods_per_year = 365 * 24  # hourly bars
            sharpe_ratio = np.sqrt(periods_per_year) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
    
    def objective(self, params_dict):
        print(f"Testing params: {params_dict}")
        train_metrics, _ = self.backtest(params_dict, self.train_data)
        print(f"Train complete: trades={train_metrics['num_trades']}, sharpe={train_metrics['sharpe_ratio']:.2f}")
        val_metrics, _ = self.backtest(params_dict, self.val_data)
        print(f"Val complete: trades={val_metrics['num_trades']}, sharpe={val_metrics['sharpe_ratio']:.2f}")
        
        # Handle edge cases
        if train_metrics['num_trades'] == 0 or val_metrics['num_trades'] == 0:
            return 999  # Bad score
        
        combined_sharpe = (train_metrics['sharpe_ratio'] * 0.7 + 
                          val_metrics['sharpe_ratio'] * 0.3)
        
        # Softer trade penalty
        trade_penalty = 0
        if train_metrics['num_trades'] < 10:
            trade_penalty = 1.0  # Fixed penalty instead of scaling
        
        score = -(combined_sharpe - trade_penalty)
        print(f"Objective score: {score:.4f}\n")
        return score
    
    def optimize(self, n_calls=50):
        print("\nStarting Bayesian Optimization...")
        
        # Only optimize parameters that the strategy actually uses
        space = [
            Integer(5, 30, name='rsi_length'),
            Integer(5, 30, name='mfi_length'),
            Integer(20, 45, name='oversold_level'),
            Integer(55, 80, name='overbought_level')
        ]
        
        @use_named_args(space)
        def objective_wrapper(**params):
            params['require_volume'] = False
            params['require_trend'] = False
            return self.objective(params)
        
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,
            acq_func='EI',
            noise=0.01,
            verbose=True,
            random_state=42
        )
        
        best_params = {
            'rsi_length': result.x[0],
            'mfi_length': result.x[1],
            'oversold_level': result.x[2],
            'overbought_level': result.x[3],
            'require_volume': False,
            'require_trend': False
        }
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {-result.fun:.4f}")
        
        return best_params, result
    
    def run_oos_test(self, params):
        print("\nRunning Out-of-Sample Test...")
        
        train_metrics, _ = self.backtest(params, self.train_data)
        val_metrics, _ = self.backtest(params, self.val_data)
        test_metrics, _ = self.backtest(params, self.test_data)
        
        print("\nBacktest Results:")
        print("="*50)
        print(f"{'Metric':<20} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*50)
        
        metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'num_trades']
        for metric in metrics:
            train_val = train_metrics[metric]
            val_val = val_metrics[metric]
            test_val = test_metrics[metric]
            
            if metric in ['total_return', 'win_rate']:
                print(f"{metric:<20} {train_val*100:>10.2f} {val_val*100:>10.2f} {test_val*100:>10.2f}")
            elif metric == 'num_trades':
                print(f"{metric:<20} {train_val:>10.0f} {val_val:>10.0f} {test_val:>10.0f}")
            else:
                print(f"{metric:<20} {train_val:>10.2f} {val_val:>10.2f} {test_val:>10.2f}")
        
        return train_metrics, val_metrics, test_metrics
    
    def save_results(self, params, metrics):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'optimization_results_{self.symbol}_{timestamp}.json'
        
        results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': datetime.now().isoformat(),
            'best_params': params,
            'metrics': metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {filename}")


def main():
    optimizer = BayesianBacktestOptimizer(
        h5_path='data/crypto_database.h5',
        symbol='SOLUSDT',
        timeframe='1_30d'
    )
    
    # Reduced n_calls for faster testing
    best_params, optimization_result = optimizer.optimize(n_calls=20)
    
    train_metrics, val_metrics, test_metrics = optimizer.run_oos_test(best_params)
    
    optimizer.save_results(
        best_params, 
        {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_result.func_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Bayesian Optimization Convergence')
    plt.grid(True)
    plt.savefig('optimization_convergence.png')


if __name__ == "__main__":
    main()