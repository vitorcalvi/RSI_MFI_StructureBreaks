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
        self.commission = 0.0002  # 0.02% (even lower for 1-minute)
        self.slippage = 0.0001    # 0.01% (minimal slippage)
        self.debug_mode = True   # Show debug info on first run
        
        # Load and prepare data
        self.data = self._load_data()
        self.train_data, self.val_data, self.test_data = self._split_data()
        
    def _load_data(self):
        """Load OHLCV data from HDF5 file"""
        path = f"{self.symbol}/{self.timeframe}"
        data = pd.read_hdf(self.h5_path, path)
        
        # Debug: Check data format
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Index: {data.index.name}")
        print(f"First few rows:\n{data.head(3)}")
        
        # Standardize column names to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Convert timestamp from nanoseconds if needed
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ns')
            data.set_index('timestamp', inplace=True)
        elif data.index.name != 'timestamp' and len(data.index) > 0:
            # If index is already datetime, keep it
            pass
            
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            
        # Check if we have enough data for meaningful analysis
        if len(data) < 1000:
            print(f"WARNING: Only {len(data)} rows of data. Need at least 1000 for proper optimization.")
            print("Available timeframes with more data:")
            print("- 1_90d (1-minute, 90 days)")
            print("- 5_90d (5-minute, 90 days)")
            print("- 15_90d (15-minute, 90 days)")
            
        # Limit to last 10000 candles for faster optimization
        return data.tail(10000).copy()
    
    def _split_data(self):
        """Split data into train/validation/test sets"""
        n = len(self.data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        return (
            self.data.iloc[:train_end].copy(),
            self.data.iloc[train_end:val_end].copy(),
            self.data.iloc[val_end:].copy()
        )
    
    def _calculate_rsi(self, prices, length):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        
        # Handle division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon
        rsi = 100 - (100 / (1 + rs))
        
        # Replace any remaining NaN/inf with 50 (neutral)
        rsi = rsi.fillna(50).replace([np.inf, -np.inf], 50)
        
        return rsi
    
    def _calculate_mfi(self, data, length):
        """Calculate Money Flow Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        pos_mf = positive_flow.rolling(window=length).sum()
        neg_mf = negative_flow.rolling(window=length).sum()
        
        # Handle division by zero
        mfr = pos_mf / (neg_mf + 1e-10)  # Add small epsilon to avoid division by zero
        mfi = 100 - (100 / (1 + mfr))
        
        # Replace any remaining NaN/inf with 50 (neutral)
        mfi = mfi.fillna(50).replace([np.inf, -np.inf], 50)
        
        return mfi
    
    def backtest(self, params, data):
        """Run backtest simulation"""
        rsi_length = int(params['rsi_length'])
        mfi_length = int(params['mfi_length'])
        oversold = params['oversold_level']
        overbought = params['overbought_level']
        
        # Calculate indicators
        rsi = self._calculate_rsi(data['close'], rsi_length)
        mfi = self._calculate_mfi(data, mfi_length)
        
        # Add trend filter (20-period SMA)
        sma20 = data['close'].rolling(window=20).mean()
        uptrend = data['close'] > sma20
        
        # Generate signals with trend filter and cooldown
        buy_signal = (rsi < oversold) & (mfi < oversold) & uptrend  # More restrictive: AND + trend
        sell_signal = (rsi > overbought) | (mfi > overbought)
        
        # Debug: Check signals (only on first run)
        if self.debug_mode:
            print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
            print(f"MFI range: {mfi.min():.2f} to {mfi.max():.2f}")
            print(f"Oversold level: {oversold}, Overbought level: {overbought}")
            print(f"Buy signals: {buy_signal.sum()}")
            print(f"Sell signals: {sell_signal.sum()}")
            self.debug_mode = False
        
        # Backtest simulation with cooldown
        position = 0
        balance = self.initial_balance
        trades = []
        equity_curve = [balance]
        last_trade_time = None
        cooldown_periods = 10  # 10-minute cooldown between trades
        
        for i in range(max(rsi_length, mfi_length, 20), len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            
            # Check cooldown
            if last_trade_time and (i - last_trade_time) < cooldown_periods:
                current_equity = balance + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                continue
            
            # Buy signal
            if buy_signal.iloc[i] and position == 0:
                position_size = (balance * 0.90) / current_price  # Use 90% instead of 95%
                cost = position_size * current_price * (1 + self.commission + self.slippage)
                
                if cost <= balance:
                    balance -= cost
                    position = position_size
                    entry_price = current_price * (1 + self.commission + self.slippage)
                    entry_time = current_time
                    last_trade_time = i
            
            # Sell signal
            elif sell_signal.iloc[i] and position > 0:
                proceeds = position * current_price * (1 - self.commission - self.slippage)
                balance += proceeds
                exit_price = current_price * (1 - self.commission - self.slippage)
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': proceeds - (position * entry_price),
                    'return_pct': (exit_price / entry_price - 1) * 100
                })
                position = 0
                last_trade_time = i
            
            # Update equity curve
            current_equity = balance + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        return trades, equity_curve
    
    def _calculate_metrics(self, trades, equity_curve):
        """Calculate performance metrics"""
        if len(trades) == 0:
            return {
                'sharpe_ratio': 0,
                'total_return': 0,
                'win_rate': 0,
                'num_trades': 0
            }
        
        # Calculate returns
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized - adjust based on timeframe)
        if self.timeframe.startswith('1_'):  # 1-minute data
            periods_per_year = 525600  # 365 * 24 * 60
        elif self.timeframe.startswith('5_'):  # 5-minute data
            periods_per_year = 105120  # 365 * 24 * 12
        elif self.timeframe.startswith('15_'):  # 15-minute data
            periods_per_year = 35040   # 365 * 24 * 4
        elif self.timeframe.startswith('30_'):  # 30-minute data
            periods_per_year = 17520   # 365 * 24 * 2
        elif 'h' in self.timeframe:  # Hourly data
            periods_per_year = 8760    # 365 * 24
        else:  # Default to 5-minute
            periods_per_year = 105120
            
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        # Other metrics
        total_return = ((equity_curve[-1] / self.initial_balance) - 1) * 100
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
    
    def objective(self, params_list):
        """Objective function for Bayesian optimization"""
        params = {
            'rsi_length': params_list[0],
            'mfi_length': params_list[1],
            'oversold_level': params_list[2],
            'overbought_level': params_list[3]
        }
        
        # Train metrics
        train_trades, train_equity = self.backtest(params, self.train_data)
        train_metrics = self._calculate_metrics(train_trades, train_equity)
        
        # Validation metrics
        val_trades, val_equity = self.backtest(params, self.val_data)
        val_metrics = self._calculate_metrics(val_trades, val_equity)
        
        # Combined score with penalty for insufficient trades (lower threshold for 1-min)
        train_score = train_metrics['sharpe_ratio']
        val_score = val_metrics['sharpe_ratio']
        
        if train_metrics['num_trades'] < 5 or val_metrics['num_trades'] < 2:  # Lower thresholds
            penalty = -1.0  # Smaller penalty
        else:
            penalty = 0
        
        # Weighted combination (70% train, 30% validation)
        combined_score = (0.7 * train_score + 0.3 * val_score) + penalty
        
        # Return negative for minimization
        return -combined_score
    
    def optimize(self, n_calls=50):
        """Run Bayesian optimization"""
        # Define search space - more conservative for 1-minute data
        space = [
            Integer(10, 25, name='rsi_length'),      # Shorter periods for 1-min
            Integer(10, 25, name='mfi_length'),      # Shorter periods for 1-min  
            Integer(15, 30, name='oversold_level'),  # More restrictive oversold
            Integer(70, 85, name='overbought_level') # More restrictive overbought
        ]
        
        print("Starting Bayesian optimization...")
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'
        )
        
        # Best parameters
        best_params = {
            'rsi_length': result.x[0],
            'mfi_length': result.x[1],
            'oversold_level': result.x[2],
            'overbought_level': result.x[3],
            'require_volume': False,
            'require_trend': True  # Now using trend filter
        }
        
        self.optimization_result = result
        return best_params
    
    def run_oos_test(self, params):
        """Run out-of-sample test"""
        # Test all three datasets
        datasets = {
            'Train': self.train_data,
            'Val': self.val_data,
            'Test': self.test_data
        }
        
        results = {}
        for name, data in datasets.items():
            trades, equity = self.backtest(params, data)
            metrics = self._calculate_metrics(trades, equity)
            results[name] = metrics
        
        return results
    
    def save_results(self, params, metrics):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{self.symbol}_{self.timeframe}_{timestamp}.json"
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': timestamp,
            'best_params': convert_types(params),
            'metrics': convert_types(metrics)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename
    
    def plot_convergence(self):
        """Plot optimization convergence"""
        if not hasattr(self, 'optimization_result'):
            print("Run optimization first")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(-np.array(self.optimization_result.func_vals))
        plt.title('Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (Sharpe Ratio)')
        plt.grid(True)
        plt.show()
    
    def generate_report(self, params):
        """Generate comprehensive performance report"""
        results = self.run_oos_test(params)
        
        print("\n" + "="*60)
        print(f"OPTIMIZATION RESULTS: {self.symbol} {self.timeframe}")
        print("="*60)
        
        print("\nBest Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)
        
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'num_trades']:
            row = f"{metric:<20}"
            for dataset in ['Train', 'Val', 'Test']:
                value = results[dataset][metric]
                if metric in ['sharpe_ratio']:
                    row += f"{value:>11.2f} "
                elif metric in ['total_return', 'win_rate']:
                    row += f"{value:>10.2f}% "
                else:
                    row += f"{value:>11.0f} "
            print(row)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BayesianBacktestOptimizer(
        h5_path="data/crypto_database.h5",
        symbol="SOLUSDT",
        timeframe="1_90d"  # Back to 1-minute data with improvements
    )
    
    # Run optimization
    best_params = optimizer.optimize(n_calls=50)
    
    # Generate report
    results = optimizer.generate_report(best_params)
    
    # Save results
    optimizer.save_results(best_params, results)
    
    # Plot convergence
    optimizer.plot_convergence()