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
        self.commission = 0.0002
        self.slippage = 0.0001
        self.debug_mode = True
        
        # Anti-overfitting parameters
        self.min_trades_per_fold = 10  # Minimum trades required
        self.max_drawdown_allowed = 0.20  # Maximum 20% drawdown
        
        # Load and prepare data
        self.data = self._load_data()
        self.train_data, self.val_data, self.test_data = self._split_data()
        
    def _load_data(self):
        """Load OHLCV data from HDF5 file"""
        path = f"{self.symbol}/{self.timeframe}"
        data = pd.read_hdf(self.h5_path, path)
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Convert timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ns')
            data.set_index('timestamp', inplace=True)
            
        return data.tail(10000).copy()
    
    def _split_data(self):
        """Split data into train/validation/test sets"""
        n = len(self.data)
        train_end = int(n * 0.6)  # Smaller training set to prevent overfitting
        val_end = int(n * 0.8)
        
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
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_mfi(self, data, length):
        """Calculate Money Flow Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        pos_mf = positive_flow.rolling(window=length).sum()
        neg_mf = negative_flow.rolling(window=length).sum()
        
        mfr = pos_mf / (neg_mf + 1e-10)
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi.fillna(50)
    
    def _calculate_volatility_filter(self, data, period=20):
        """Calculate volatility filter using ATR"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # High volatility when ATR is above its moving average
        atr_ma = atr.rolling(window=period).mean()
        return atr > atr_ma
    
    def backtest(self, params, data):
        """Run backtest with anti-overfitting measures"""
        rsi_length = int(params['rsi_length'])
        mfi_length = int(params['mfi_length'])
        oversold = params['oversold_level']
        overbought = params['overbought_level']
        
        # Calculate indicators
        rsi = self._calculate_rsi(data['close'], rsi_length)
        mfi = self._calculate_mfi(data, mfi_length)
        
        # Volatility filter
        high_volatility = self._calculate_volatility_filter(data)
        
        # EMA trend filter (more responsive than SMA)
        ema_short = data['close'].ewm(span=20, adjust=False).mean()
        ema_long = data['close'].ewm(span=50, adjust=False).mean()
        uptrend = ema_short > ema_long
        
        # More selective entry conditions
        rsi_oversold = rsi < oversold
        mfi_oversold = mfi < oversold
        
        # Require at least one indicator to be oversold AND volatility not too high
        buy_signal = ((rsi_oversold | mfi_oversold) & 
                     uptrend & 
                     ~high_volatility)  # Avoid high volatility periods
        
        # Exit conditions
        rsi_overbought = rsi > overbought
        mfi_overbought = mfi > overbought
        
        sell_signal = (rsi_overbought | mfi_overbought | ~uptrend)
        
        # Debug info
        if self.debug_mode:
            print(f"\nBacktest Debug Info:")
            print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
            print(f"MFI range: {mfi.min():.2f} to {mfi.max():.2f}")
            print(f"Parameters: oversold={oversold}, overbought={overbought}")
            print(f"Buy signals: {buy_signal.sum()}")
            print(f"Sell signals: {sell_signal.sum()}")
            self.debug_mode = False
        
        # Backtest with position sizing and risk management
        position = 0
        balance = self.initial_balance
        trades = []
        equity_curve = [balance]
        max_equity = balance
        
        for i in range(max(rsi_length, mfi_length, 50), len(data)):
            current_price = data['close'].iloc[i]
            
            # Calculate current equity and drawdown
            current_equity = balance + (position * current_price if position > 0 else 0)
            max_equity = max(max_equity, current_equity)
            drawdown = (max_equity - current_equity) / max_equity
            
            # Stop trading if drawdown exceeds limit
            if drawdown > self.max_drawdown_allowed:
                if position > 0:
                    # Close position
                    proceeds = position * current_price * (1 - self.commission - self.slippage)
                    balance += proceeds
                    position = 0
                equity_curve.append(balance)
                continue
            
            # Buy signal
            if buy_signal.iloc[i] and position == 0:
                # Risk-based position sizing
                risk_amount = balance * 0.02  # Risk 2% per trade
                position_size = (balance * 0.95) / current_price
                cost = position_size * current_price * (1 + self.commission + self.slippage)
                
                if cost <= balance:
                    balance -= cost
                    position = position_size
                    entry_price = current_price * (1 + self.commission + self.slippage)
                    entry_time = data.index[i]
                    stop_loss = entry_price * 0.98  # 2% stop loss
            
            # Sell signal or stop loss
            elif position > 0:
                if sell_signal.iloc[i] or current_price <= stop_loss:
                    proceeds = position * current_price * (1 - self.commission - self.slippage)
                    balance += proceeds
                    exit_price = current_price * (1 - self.commission - self.slippage)
                    
                    # Record trade
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': proceeds - (position * entry_price),
                        'return_pct': (exit_price / entry_price - 1) * 100
                    })
                    position = 0
            
            equity_curve.append(current_equity)
        
        return trades, equity_curve
    
    def _calculate_metrics(self, trades, equity_curve):
        """Calculate robust performance metrics"""
        if len(trades) < 5:  # Require minimum trades
            return {
                'sharpe_ratio': -5,  # Heavy penalty
                'sortino_ratio': -5,
                'calmar_ratio': -5,
                'total_return': -10,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 100,
                'num_trades': len(trades)
            }
        
        # Calculate returns
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Annualization factor
        if self.timeframe.startswith('5_'):
            periods_per_year = 105120  # 5-minute bars
        else:
            periods_per_year = 525600  # 1-minute bars
            
        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
            
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
        else:
            sortino_ratio = sharpe_ratio
            
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Calmar ratio
        total_return = ((equity_curve[-1] / self.initial_balance) - 1) * 100
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate and profit factor
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100
        
        if losing_trades:
            gross_profits = sum(t['pnl'] for t in winning_trades)
            gross_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        else:
            profit_factor = float('inf') if winning_trades else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades)
        }
    
    def objective(self, params_list):
        """Multi-objective function with regularization"""
        params = {
            'rsi_length': params_list[0],
            'mfi_length': params_list[1],
            'oversold_level': params_list[2],
            'overbought_level': params_list[3]
        }
        
        # Walk-forward analysis on training data
        train_splits = 3
        fold_size = len(self.train_data) // train_splits
        fold_metrics = []
        
        for i in range(train_splits - 1):
            # Train on earlier data, validate on later data
            train_fold = self.train_data.iloc[i*fold_size:(i+1)*fold_size]
            val_fold = self.train_data.iloc[(i+1)*fold_size:(i+2)*fold_size]
            
            # Get metrics for validation fold
            trades, equity = self.backtest(params, val_fold)
            metrics = self._calculate_metrics(trades, equity)
            fold_metrics.append(metrics)
        
        # Validation set metrics
        val_trades, val_equity = self.backtest(params, self.val_data)
        val_metrics = self._calculate_metrics(val_trades, val_equity)
        
        # Compute robust score
        if fold_metrics and val_metrics['num_trades'] >= 5:
            # Average metrics across folds
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in fold_metrics])
            avg_sortino = np.mean([m['sortino_ratio'] for m in fold_metrics])
            avg_calmar = np.mean([m['calmar_ratio'] for m in fold_metrics])
            
            # Combine multiple metrics with weights
            train_score = (
                0.4 * avg_sharpe +
                0.3 * avg_sortino +
                0.3 * avg_calmar
            )
            
            val_score = (
                0.4 * val_metrics['sharpe_ratio'] +
                0.3 * val_metrics['sortino_ratio'] +
                0.3 * val_metrics['calmar_ratio']
            )
            
            # Weighted combination favoring validation
            combined_score = 0.4 * train_score + 0.6 * val_score
            
            # Regularization penalties
            if val_metrics['max_drawdown'] > 15:  # Penalize high drawdown
                combined_score *= 0.8
            if val_metrics['profit_factor'] < 1.2:  # Penalize low profit factor
                combined_score *= 0.9
                
        else:
            combined_score = -10  # Heavy penalty for insufficient trades
        
        return -combined_score  # Negative for minimization
    
    def optimize(self, n_calls=50):
        """Run Bayesian optimization with constraints"""
        # More conservative search space
        space = [
            Integer(14, 28, name='rsi_length'),      # Standard RSI periods
            Integer(14, 28, name='mfi_length'),      # Standard MFI periods
            Integer(20, 30, name='oversold_level'),  # Conservative oversold
            Integer(70, 80, name='overbought_level') # Conservative overbought
        ]
        
        print("\nStarting robust Bayesian optimization...")
        print("Using walk-forward validation and multiple performance metrics")
        
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,  # More random exploration
            random_state=42,
            acq_func='EI'
        )
        
        # Best parameters
        best_params = {
            'rsi_length': result.x[0],
            'mfi_length': result.x[1],
            'oversold_level': result.x[2],
            'overbought_level': result.x[3]
        }
        
        self.optimization_result = result
        return best_params
    
    def run_oos_test(self, params):
        """Run out-of-sample test"""
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
        
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj == float('inf'):
                return 999999
            return obj
        
        results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': timestamp,
            'best_params': convert_types(params),
            'metrics': convert_types(metrics),
            'optimization_details': {
                'min_trades_per_fold': self.min_trades_per_fold,
                'max_drawdown_allowed': self.max_drawdown_allowed,
                'train_size': len(self.train_data),
                'val_size': len(self.val_data),
                'test_size': len(self.test_data)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
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
        plt.ylabel('Combined Score')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_equity_curves(self, params):
        """Plot equity curves for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        datasets = {
            'Train': self.train_data,
            'Validation': self.val_data,
            'Test': self.test_data
        }
        
        for ax, (name, data) in zip(axes, datasets.items()):
            trades, equity = self.backtest(params, data)
            ax.plot(equity)
            ax.set_title(f'{name} Equity Curve')
            ax.set_xlabel('Time')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, params):
        """Generate comprehensive performance report"""
        results = self.run_oos_test(params)
        
        print("\n" + "="*70)
        print(f"ROBUST OPTIMIZATION RESULTS: {self.symbol} {self.timeframe}")
        print("="*70)
        
        print("\nBest Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 70)
        
        metrics_to_show = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                          'total_return', 'win_rate', 'profit_factor', 
                          'max_drawdown', 'num_trades']
        
        for metric in metrics_to_show:
            row = f"{metric:<20}"
            for dataset in ['Train', 'Val', 'Test']:
                if metric in results[dataset]:
                    value = results[dataset][metric]
                    if metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor']:
                        row += f"{value:>11.2f} "
                    elif metric in ['total_return', 'win_rate', 'max_drawdown']:
                        row += f"{value:>10.1f}% "
                    else:
                        row += f"{value:>11.0f} "
                else:
                    row += f"{'N/A':>11} "
            print(row)
        
        # Calculate consistency score
        if results['Test']['num_trades'] >= 5:
            consistency = 1 - abs(results['Val']['sharpe_ratio'] - results['Test']['sharpe_ratio']) / max(abs(results['Val']['sharpe_ratio']), 1)
            print(f"\nConsistency Score: {consistency:.2%}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BayesianBacktestOptimizer(
        h5_path="data/crypto_database.h5",
        symbol="SOLUSDT",
        timeframe="5_90d"
    )
    
    # Run optimization
    best_params = optimizer.optimize(n_calls=50)
    
    # Generate report
    results = optimizer.generate_report(best_params)
    
    # Save results
    optimizer.save_results(best_params, results)
    
    # Plot convergence
    optimizer.plot_convergence()
    
    # Plot equity curves
    optimizer.plot_equity_curves(best_params)