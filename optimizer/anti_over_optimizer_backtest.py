import numpy as np
import pandas as pd
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer, Real
import json
import warnings
warnings.filterwarnings('ignore')

class ImprovedCryptoOptimizer:
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
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Convert timestamp if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ns')
            data.set_index('timestamp', inplace=True)
            
        # Use last 10,000 candles
        data = data.tail(10000).copy()
        print(f"Using last {len(data)} candles")
        
        # Add market regime detection
        sma_200 = data['close'].rolling(200).mean()
        data['bull_market'] = data['close'] > sma_200
        
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
    
    def backtest_improved(self, params, data):
        """Improved backtest with dynamic thresholds and better exit logic"""
        rsi_length, mfi_length, oversold_base, overbought_base, stop_loss_pct, take_profit_pct = params
        
        # Calculate indicators
        rsi = self.calculate_rsi(data['close'], int(rsi_length))
        mfi = self.calculate_mfi(data['high'], data['low'], 
                                data['close'], data['volume'], int(mfi_length))
        
        # Dynamic thresholds based on market regime
        oversold = np.where(data['bull_market'], oversold_base + 10, oversold_base)
        overbought = np.where(data['bull_market'], overbought_base + 10, overbought_base)
        
        # More flexible entry signals (removed SMA requirement)
        rsi_oversold = rsi < oversold
        mfi_oversold = mfi < oversold
        
        # Entry: Either RSI or MFI oversold (not both required)
        buy_signal = rsi_oversold | mfi_oversold
        
        # Exit signals
        rsi_overbought = rsi > overbought
        mfi_overbought = mfi > overbought
        
        # Initialize trading variables
        position = 0
        entry_price = 0
        balance = self.initial_balance
        trades = []
        equity_curve = [self.initial_balance]
        
        for i in range(max(int(rsi_length), int(mfi_length)) + 1, len(data)):
            # Calculate current equity
            current_equity = balance if position == 0 else position * data['close'].iloc[i]
            equity_curve.append(current_equity)
            
            if position > 0:
                # Check stop loss
                if data['close'].iloc[i] < entry_price * (1 - stop_loss_pct):
                    sell_price = data['close'].iloc[i]
                    balance = position * sell_price * (1 - self.commission)
                    trades.append({
                        'entry': entry_price,
                        'exit': sell_price,
                        'return': (sell_price / entry_price) - 1,
                        'type': 'stop_loss'
                    })
                    position = 0
                    entry_price = 0
                    
                # Check take profit
                elif data['close'].iloc[i] > entry_price * (1 + take_profit_pct):
                    sell_price = data['close'].iloc[i]
                    balance = position * sell_price * (1 - self.commission)
                    trades.append({
                        'entry': entry_price,
                        'exit': sell_price,
                        'return': (sell_price / entry_price) - 1,
                        'type': 'take_profit'
                    })
                    position = 0
                    entry_price = 0
                    
                # Check overbought exit
                elif rsi_overbought[i] or mfi_overbought[i]:
                    sell_price = data['close'].iloc[i]
                    balance = position * sell_price * (1 - self.commission)
                    trades.append({
                        'entry': entry_price,
                        'exit': sell_price,
                        'return': (sell_price / entry_price) - 1,
                        'type': 'signal_exit'
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
                'return': (final_price / entry_price) - 1,
                'type': 'final_close'
            })
            
        # Final equity
        final_equity = balance if position == 0 else position * data['close'].iloc[-1]
        equity_curve.append(final_equity)
        
        # Calculate metrics
        if len(trades) < 5:
            return {'score': -10.0, 'trades': 0}
        
        # Calculate returns from equity curve
        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio
        sharpe = self.calculate_sharpe(daily_returns)
        
        # Win rate
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / len(trades)
        
        # Profit factor
        gross_profits = sum([t['return'] for t in trades if t['return'] > 0])
        gross_losses = abs(sum([t['return'] for t in trades if t['return'] < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        
        # Total return
        total_return = (final_equity / self.initial_balance) - 1
        
        # Score combining multiple metrics
        score = (
            0.3 * sharpe +
            0.3 * total_return * 10 +  # Scale total return
            0.2 * win_rate * 5 +        # Scale win rate
            0.2 * (profit_factor - 1)   # Profit factor above 1
        )
        
        return {
            'score': score,
            'trades': len(trades),
            'sharpe': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def calculate_sharpe(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return -10.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * mean_return / std_return
        return sharpe
    
    def objective(self, params):
        """Objective function for optimization"""
        self.call_count += 1
        
        # Evaluate on train and validation
        train_result = self.backtest_improved(params, self.train)
        val_result = self.backtest_improved(params, self.val)
        
        # Weighted score (prioritize validation)
        score = 0.3 * train_result['score'] + 0.7 * val_result['score']
        
        # Print progress every 10 calls
        if self.call_count % 10 == 0:
            print(f"Call {self.call_count}: Train score={train_result['score']:.3f}, "
                  f"Val score={val_result['score']:.3f}, Combined={score:.3f}")
        
        return -score  # Minimize negative score
    
    def optimize(self, n_calls=50):
        """Run Bayesian optimization with improved parameter space"""
        space = [
            Integer(10, 20, name='rsi_length'),          # Shorter periods for crypto
            Integer(10, 20, name='mfi_length'),
            Integer(10, 30, name='oversold_base'),       # More extreme for crypto
            Integer(70, 90, name='overbought_base'),
            Real(0.02, 0.05, name='stop_loss_pct'),      # 2-5% stop loss
            Real(0.03, 0.10, name='take_profit_pct')     # 3-10% take profit
        ]
        
        print(f"\nStarting IMPROVED optimization for {self.symbol} {self.timeframe}...")
        print(f"Number of optimization calls: {n_calls}")
        
        self.call_count = 0
        
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=15,
            random_state=42,
            acq_func='EI'
        )
        
        best_params = result.x
        print(f"\nOptimization complete! Total calls: {self.call_count}")
        print(f"Best parameters found:")
        print(f"RSI Length: {best_params[0]}")
        print(f"MFI Length: {best_params[1]}")
        print(f"Oversold Base: {best_params[2]}")
        print(f"Overbought Base: {best_params[3]}")
        print(f"Stop Loss: {best_params[4]*100:.1f}%")
        print(f"Take Profit: {best_params[5]*100:.1f}%")
        
        return best_params
    
    def evaluate_detailed(self, params):
        """Detailed evaluation with all metrics"""
        print(f"\nEvaluating final parameters...")
        
        # Get detailed metrics for each dataset
        train_result = self.backtest_improved(params, self.train)
        val_result = self.backtest_improved(params, self.val)
        test_result = self.backtest_improved(params, self.test)
        
        # Print results
        print(f"\nDETAILED PERFORMANCE RESULTS:")
        print(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)
        
        metrics = ['sharpe', 'total_return', 'win_rate', 'profit_factor', 'trades']
        
        for metric in metrics:
            if metric in ['total_return', 'win_rate']:
                print(f"{metric:<20} {train_result.get(metric, 0):<12.1%} "
                      f"{val_result.get(metric, 0):<12.1%} "
                      f"{test_result.get(metric, 0):<12.1%}")
            elif metric == 'trades':
                print(f"{metric:<20} {train_result.get(metric, 0):<12.0f} "
                      f"{val_result.get(metric, 0):<12.0f} "
                      f"{test_result.get(metric, 0):<12.0f}")
            else:
                print(f"{metric:<20} {train_result.get(metric, 0):<12.3f} "
                      f"{val_result.get(metric, 0):<12.3f} "
                      f"{test_result.get(metric, 0):<12.3f}")
        
        # Trading analysis
        self.analyze_trades(params, self.train)
        
        return {
            'train': train_result,
            'val': val_result,
            'test': test_result
        }
    
    def analyze_trades(self, params, data):
        """Analyze trade distribution"""
        print(f"\nTRADE ANALYSIS:")
        
        # Run backtest to get trades
        result = self.backtest_improved(params, data)
        
        # This would need the trades from backtest
        print(f"Total trades: {result['trades']}")
        print(f"Avg trades per month: {result['trades'] / (len(data) / (252*5/12)):.1f}")
        
    def save_results(self, params, results):
        """Save results to JSON file"""
        output = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'best_params': {
                'rsi_length': int(params[0]),
                'mfi_length': int(params[1]),
                'oversold_base': int(params[2]),
                'overbought_base': int(params[3]),
                'stop_loss_pct': float(params[4]),
                'take_profit_pct': float(params[5])
            },
            'performance': results,
            'improvements': [
                'Removed restrictive SMA filter',
                'Dynamic thresholds based on market regime',
                'Added take profit targets',
                'Adjusted parameters for crypto volatility',
                'More flexible entry conditions'
            ]
        }
        
        filename = f"improved_{self.symbol}_{self.timeframe}_{output['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ImprovedCryptoOptimizer(
        h5_path="data/crypto_database.h5",
        symbol="SOLUSDT",
        timeframe="5_90d"
    )
    
    # Run optimization
    best_params = optimizer.optimize(n_calls=250)
    
    # Evaluate on all datasets
    results = optimizer.evaluate_detailed(best_params)
    
    # Save results
    optimizer.save_results(best_params, results)