import numpy as np
import pandas as pd
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer
import json
import warnings
warnings.filterwarnings('ignore')

class BayesianBacktestOptimizer:
    def __init__(self, h5_path, symbol, timeframe, initial_balance=10000):
        self.h5_path = h5_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.commission = 0.0002  # 0.02%
        self.data = self.load_data()
        self.train, self.val, self.test = self.split_data()
        self.call_count = 0
        
    def load_data(self):
        """Load OHLCV data from HDF5 file"""
        path = f"{self.symbol}/{self.timeframe}"
        
        data = pd.read_hdf(self.h5_path, path)
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Convert timestamp if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ns')
            data.set_index('timestamp', inplace=True)
            
        # Use last 10,000 candles
        data = data.tail(10000).copy()
        print(f"Using last {len(data)} candles")
        
        return data
    
    def split_data(self):
        """Split data into train (60%), val (20%), test (20%)"""
        n = len(self.data)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        train = self.data.iloc[:train_end].copy()
        val = self.data.iloc[train_end:val_end].copy()
        test = self.data.iloc[val_end:].copy()
        
        print(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def calculate_rsi(self, close, period):
        """Calculate RSI indicator"""
        delta = close.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def calculate_mfi(self, high, low, close, volume, period):
        """Calculate MFI indicator"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0)
        negative_flow = money_flow.where(typical_price.diff() < 0, 0)
        
        sum_positive = positive_flow.rolling(period).sum()
        sum_negative = negative_flow.rolling(period).sum()
        
        sum_negative = sum_negative.replace(0, 1e-10)
        
        mfi_ratio = sum_positive / sum_negative
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        return mfi.fillna(50)
    
    def backtest_simple(self, params, data):
        """Simplified backtest that returns only key metrics"""
        rsi_length, mfi_length, oversold, overbought = params
        
        # Calculate indicators
        rsi = self.calculate_rsi(data['close'], rsi_length)
        mfi = self.calculate_mfi(data['high'], data['low'], 
                                data['close'], data['volume'], mfi_length)
        sma_50 = data['close'].rolling(50).mean()
        
        # Generate signals
        buy_signal = ((rsi < oversold) | (mfi < oversold)) & (data['close'] > sma_50)
        
        # Fast vectorized backtest
        position = 0
        entry_price = 0
        balance = self.initial_balance
        num_trades = 0
        returns = []
        
        for i in range(51, len(data)):
            # Check stop loss
            if position > 0 and data['close'].iloc[i] < entry_price * 0.98:
                sell_price = data['close'].iloc[i]
                trade_return = (sell_price / entry_price) - 1
                returns.append(trade_return)
                balance = position * sell_price * (1 - self.commission)
                position = 0
                entry_price = 0
                num_trades += 1
                
            # Check sell signal
            elif position > 0:
                sell_cond = ((rsi.iloc[i] > overbought) | 
                           (mfi.iloc[i] > overbought) | 
                           (data['close'].iloc[i] < sma_50.iloc[i]))
                
                if sell_cond:
                    sell_price = data['close'].iloc[i]
                    trade_return = (sell_price / entry_price) - 1
                    returns.append(trade_return)
                    balance = position * sell_price * (1 - self.commission)
                    position = 0
                    entry_price = 0
                    num_trades += 1
            
            # Check buy signal
            elif position == 0 and buy_signal.iloc[i]:
                entry_price = data['close'].iloc[i]
                position = (balance * 0.95) / entry_price * (1 - self.commission)
                balance = 0
        
        # Close any open position
        if position > 0:
            final_price = data['close'].iloc[-1]
            trade_return = (final_price / entry_price) - 1
            returns.append(trade_return)
            num_trades += 1
        
        # Calculate quick metrics
        if num_trades < 10:
            return -1.0  # Penalty for insufficient trades
        
        returns_series = pd.Series(returns)
        sharpe = self.calculate_sharpe(returns_series)
        
        return sharpe
    
    def calculate_sharpe(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return -1.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * mean_return / std_return
        return sharpe
    
    def objective(self, params):
        """Simplified objective function"""
        self.call_count += 1
        
        # Quick evaluation on train and validation
        train_sharpe = self.backtest_simple(params, self.train)
        val_sharpe = self.backtest_simple(params, self.val)
        
        # Weighted score (prioritize validation)
        score = 0.3 * train_sharpe + 0.7 * val_sharpe
        
        # Print progress every 10 calls
        if self.call_count % 10 == 0:
            print(f"Call {self.call_count}: Train Sharpe={train_sharpe:.3f}, "
                  f"Val Sharpe={val_sharpe:.3f}, Score={score:.3f}")
        
        return -score  # Minimize negative score
    
    def backtest_detailed(self, params, data):
        """Detailed backtest with all metrics (only for final evaluation)"""
        rsi_length, mfi_length, oversold, overbought = params
        
        # Calculate indicators
        rsi = self.calculate_rsi(data['close'], rsi_length)
        mfi = self.calculate_mfi(data['high'], data['low'], 
                                data['close'], data['volume'], mfi_length)
        sma_50 = data['close'].rolling(50).mean()
        
        # Generate signals
        buy_signal = ((rsi < oversold) | (mfi < oversold)) & (data['close'] > sma_50)
        
        # Initialize trading variables
        position = 0
        entry_price = 0
        balance = self.initial_balance
        trades = []
        equity_curve = [self.initial_balance]
        
        for i in range(51, len(data)):
            current_equity = balance if position == 0 else position * data['close'].iloc[i]
            equity_curve.append(current_equity)
            
            # Check stop loss
            if position > 0 and data['close'].iloc[i] < entry_price * 0.98:
                sell_price = data['close'].iloc[i]
                balance = position * sell_price * (1 - self.commission)
                trades.append({
                    'entry': entry_price,
                    'exit': sell_price,
                    'return': (sell_price / entry_price) - 1
                })
                position = 0
                entry_price = 0
                
            # Check sell signal
            elif position > 0:
                sell_cond = ((rsi.iloc[i] > overbought) | 
                           (mfi.iloc[i] > overbought) | 
                           (data['close'].iloc[i] < sma_50.iloc[i]))
                
                if sell_cond:
                    sell_price = data['close'].iloc[i]
                    balance = position * sell_price * (1 - self.commission)
                    trades.append({
                        'entry': entry_price,
                        'exit': sell_price,
                        'return': (sell_price / entry_price) - 1
                    })
                    position = 0
                    entry_price = 0
            
            # Check buy signal
            elif position == 0 and buy_signal.iloc[i]:
                entry_price = data['close'].iloc[i]
                position = (balance * 0.95) / entry_price * (1 - self.commission)
                balance = 0
        
        # Close any open position
        if position > 0:
            final_price = data['close'].iloc[-1]
            balance = position * final_price * (1 - self.commission)
            trades.append({
                'entry': entry_price,
                'exit': final_price,
                'return': (final_price / entry_price) - 1
            })
            
        # Final equity
        final_equity = balance if position == 0 else position * data['close'].iloc[-1]
        
        # Calculate all metrics
        if len(trades) == 0:
            return {
                'sharpe': -10.0,
                'sortino': -10.0,
                'total_return': -1.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 1.0
            }
        
        returns = pd.Series([t['return'] for t in trades])
        equity_series = pd.Series(equity_curve)
        
        # Metrics
        sharpe = self.calculate_sharpe(returns)
        winning_trades = len(returns[returns > 0])
        win_rate = winning_trades / len(trades)
        
        # Sortino
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = sharpe * 2
        
        # Max drawdown
        cumulative = equity_series / equity_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Total return
        total_return = (final_equity / self.initial_balance) - 1
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }
    
    def optimize(self, n_calls=50):
        """Run Bayesian optimization"""
        space = [
            Integer(14, 28, name='rsi_length'),
            Integer(14, 28, name='mfi_length'),
            Integer(25, 35, name='oversold_level'),
            Integer(65, 75, name='overbought_level')
        ]
        
        print(f"\nStarting optimization for {self.symbol} {self.timeframe}...")
        print(f"Number of optimization calls: {n_calls}")
        
        self.call_count = 0
        
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,
            random_state=42,
            acq_func='EI'
        )
        
        best_params = result.x
        print(f"\nOptimization complete! Total calls: {self.call_count}")
        print(f"Best parameters found:")
        print(f"RSI Length: {best_params[0]}")
        print(f"MFI Length: {best_params[1]}")
        print(f"Oversold Level: {best_params[2]}")
        print(f"Overbought Level: {best_params[3]}")
        
        return best_params
    
    def evaluate(self, params):
        """Evaluate parameters on all datasets"""
        print(f"\nEvaluating final parameters...")
        
        # Get detailed metrics for each dataset
        train_metrics = self.backtest_detailed(params, self.train)
        val_metrics = self.backtest_detailed(params, self.val)
        test_metrics = self.backtest_detailed(params, self.test)
        
        # Print results
        print(f"\nDETAILED PERFORMANCE RESULTS:")
        print(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)
        
        metrics_list = ['sharpe', 'sortino', 'total_return', 'num_trades', 'win_rate', 'max_drawdown']
        
        for metric in metrics_list:
            train_val = train_metrics[metric]
            val_val = val_metrics[metric]
            test_val = test_metrics[metric]
            
            if metric in ['total_return', 'win_rate', 'max_drawdown']:
                print(f"{metric:<20} {train_val:<12.1%} {val_val:<12.1%} {test_val:<12.1%}")
            elif metric == 'num_trades':
                print(f"{metric:<20} {train_val:<12.0f} {val_val:<12.0f} {test_val:<12.0f}")
            else:
                print(f"{metric:<20} {train_val:<12.3f} {val_val:<12.3f} {test_val:<12.3f}")
        
        # Consistency check
        if val_metrics['sharpe'] < -0.5 and train_metrics['sharpe'] > 0.5:
            print("\n⚠️  WARNING: Large discrepancy between train and validation performance!")
            print("This suggests overfitting. Consider:")
            print("- Using more conservative parameters")
            print("- Adding more data")
            print("- Simplifying the strategy")
        
        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
    
    def save_results(self, params, metrics):
        """Save results to JSON file"""
        results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'best_params': {
                'rsi_length': int(params[0]),
                'mfi_length': int(params[1]),
                'oversold_level': int(params[2]),
                'overbought_level': int(params[3])
            },
            'metrics': metrics
        }
        
        filename = f"optimized_{self.symbol}_{self.timeframe}_{results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BayesianBacktestOptimizer(
        h5_path="data/crypto_database.h5",
        symbol="SOLUSDT",
        timeframe="5_90d"
    )
    
    # Run optimization with fewer calls for speed
    best_params = optimizer.optimize(n_calls=50)
    
    # Evaluate on all datasets
    metrics = optimizer.evaluate(best_params)
    
    # Save results
    optimizer.save_results(best_params, metrics)