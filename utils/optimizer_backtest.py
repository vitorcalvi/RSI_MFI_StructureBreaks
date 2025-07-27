import pandas as pd
import numpy as np
import h5py
import json
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer, Real
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BayesianBacktestOptimizer:
    def __init__(self, h5_path, symbol, timeframe, initial_balance=10000):
        self.h5_path = h5_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.commission = 0.0005  # Realistic 0.05%
        self.slippage = 0.0003    # Realistic 0.03%
        
        # Load and prepare data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load OHLCV data from HDF5 file"""
        path = f"{self.symbol}/{self.timeframe}"
        data = pd.read_hdf(self.h5_path, path)
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Need minimum 6 months of data for proper validation
        min_required = 130000 if self.timeframe.startswith('1_') else 26000  # 6 months
        if len(data) < min_required:
            print(f"ERROR: Need at least {min_required} rows, got {len(data)}")
            print("Use longer timeframe or more data")
            raise ValueError("Insufficient data for proper backtesting")
            
        return data
    
    def _calculate_rsi(self, prices, length):
        """Calculate RSI with proper error handling"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=length, min_periods=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length, min_periods=length).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def backtest(self, params, data):
        """Simplified backtest with fewer parameters"""
        rsi_length = int(params['rsi_length'])
        rsi_oversold = params['rsi_oversold']
        rsi_overbought = params['rsi_overbought']
        
        # Calculate indicators
        rsi = self._calculate_rsi(data['close'], rsi_length)
        sma50 = data['close'].rolling(window=50).mean()
        
        # Simple signals with trend filter
        uptrend = data['close'] > sma50
        buy_signal = (rsi < rsi_oversold) & uptrend
        sell_signal = (rsi > rsi_overbought)
        
        # Backtest simulation
        position = 0
        balance = self.initial_balance
        trades = []
        equity_curve = [balance]
        last_trade_idx = -10  # Cooldown
        
        for i in range(50, len(data)):  # Start after SMA period
            current_price = data['close'].iloc[i]
            
            # Cooldown between trades
            if i - last_trade_idx < 5:
                equity_curve.append(balance + (position * current_price if position > 0 else 0))
                continue
            
            # Buy signal
            if buy_signal.iloc[i] and position == 0:
                position_size = (balance * 0.95) / current_price
                cost = position_size * current_price * (1 + self.commission + self.slippage)
                
                if cost <= balance:
                    balance -= cost
                    position = position_size
                    entry_price = current_price * (1 + self.commission + self.slippage)
                    entry_time = data.index[i]
                    last_trade_idx = i
            
            # Sell signal
            elif sell_signal.iloc[i] and position > 0:
                proceeds = position * current_price * (1 - self.commission - self.slippage)
                balance += proceeds
                exit_price = current_price * (1 - self.commission - self.slippage)
                
                pnl = proceeds - (position * entry_price)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': (exit_price / entry_price - 1) * 100
                })
                position = 0
                last_trade_idx = i
            
            # Update equity
            current_equity = balance + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        return trades, equity_curve
    
    def _calculate_metrics(self, trades, equity_curve):
        """Calculate performance metrics"""
        if len(trades) == 0:
            return {'sharpe_ratio': 0, 'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        periods_per_year = 525600 if self.timeframe.startswith('1_') else 105120
        sharpe_ratio = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(periods_per_year)
        
        total_return = ((equity_curve[-1] / self.initial_balance) - 1) * 100
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
    
    def walk_forward_test(self, params, train_size=0.7, step_size=0.1):
        """Walk-forward validation - core fix"""
        n = len(self.data)
        train_length = int(n * train_size)
        step_length = int(n * step_size)
        
        all_results = []
        start_idx = 0
        
        while start_idx + train_length + step_length <= n:
            end_train = start_idx + train_length
            end_test = min(end_train + step_length, n)
            
            # Get data windows
            train_data = self.data.iloc[start_idx:end_train]
            test_data = self.data.iloc[end_train:end_test]
            
            if len(test_data) < 100:  # Skip if test period too short
                break
                
            # Run backtest on test period
            trades, equity = self.backtest(params, test_data)
            metrics = self._calculate_metrics(trades, equity)
            
            all_results.append(metrics)
            start_idx += step_length
        
        if not all_results:
            return {'sharpe_ratio': 0, 'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        # Average results across all walk-forward periods
        avg_metrics = {}
        for key in all_results[0].keys():
            avg_metrics[key] = np.mean([r[key] for r in all_results])
        
        return avg_metrics
    
    def objective(self, params_list):
        """Objective function for optimization"""
        params = {
            'rsi_length': params_list[0],
            'rsi_oversold': params_list[1], 
            'rsi_overbought': params_list[2]
        }
        
        # Use walk-forward validation instead of simple train/val split
        metrics = self.walk_forward_test(params)
        
        # Penalty for insufficient trades
        if metrics['num_trades'] < 10:
            penalty = -2.0
        else:
            penalty = 0
        
        score = metrics['sharpe_ratio'] + penalty
        return -score  # Negative for minimization
    
    def optimize(self, n_calls=30):
        """Run Bayesian optimization with fewer parameters"""
        # Reduced parameter space to prevent overfitting
        space = [
            Integer(14, 21, name='rsi_length'),     # RSI period
            Integer(20, 35, name='rsi_oversold'),   # Oversold level
            Integer(65, 80, name='rsi_overbought')  # Overbought level
        ]
        
        print("Starting optimization with walk-forward validation...")
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'
        )
        
        best_params = {
            'rsi_length': result.x[0],
            'rsi_oversold': result.x[1],
            'rsi_overbought': result.x[2]
        }
        
        self.optimization_result = result
        return best_params
    
    def final_test(self, params):
        """Final out-of-sample test on last 20% of data"""
        split_point = int(len(self.data) * 0.8)
        test_data = self.data.iloc[split_point:]
        
        trades, equity = self.backtest(params, test_data)
        return self._calculate_metrics(trades, equity)
    
    def generate_report(self, params):
        """Generate performance report"""
        wf_metrics = self.walk_forward_test(params)
        final_metrics = self.final_test(params)
        
        print("\n" + "="*50)
        print(f"RESULTS: {self.symbol} {self.timeframe}")
        print("="*50)
        
        print("\nOptimal Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'Metric':<20} {'Walk-Forward':<15} {'Final Test':<15}")
        print("-" * 50)
        
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'num_trades']:
            wf_val = wf_metrics[metric]
            final_val = final_metrics[metric]
            
            if metric in ['sharpe_ratio']:
                print(f"{metric:<20} {wf_val:>14.2f} {final_val:>14.2f}")
            elif metric in ['total_return', 'win_rate']:
                print(f"{metric:<20} {wf_val:>13.1f}% {final_val:>13.1f}%")
            else:
                print(f"{metric:<20} {wf_val:>14.0f} {final_val:>14.0f}")
        
        return {'walk_forward': wf_metrics, 'final_test': final_metrics}

# Usage
if __name__ == "__main__":
    try:
        optimizer = BayesianBacktestOptimizer(
            h5_path="data/crypto_database.h5",
            symbol="SOLUSDT", 
            timeframe="5_90d"  # Use 5-minute for faster testing
        )
        
        best_params = optimizer.optimize(n_calls=30)
        results = optimizer.generate_report(best_params)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check your data file and timeframe")