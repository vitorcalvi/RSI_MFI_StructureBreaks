import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import json
import warnings
import os
warnings.filterwarnings('ignore')

class AdvancedCryptoHFTOptimizer:
    def __init__(self, h5_path=None, symbol="ZORAUSDT", timeframe="5m", initial_balance=10000):
        self.h5_path = h5_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        
        # Realistic HFT fees for crypto
        self.taker_fee = 0.001    # 0.1% taker fee
        self.maker_fee = 0.001    # 0.1% maker fee
        self.slippage = 0.0005    # 0.05% slippage
        
        # Load or generate data
        self.data = self.load_or_generate_data()
        self.prepare_data()
        
        # Optimization tracking
        self.call_count = 0
        self.best_results = []
        
    def load_or_generate_data(self):
        """Load data from H5 or generate synthetic data"""
        if self.h5_path and os.path.exists(self.h5_path):
            try:
                return self.load_h5_data()
            except Exception as e:
                print(f"H5 loading failed: {e}. Generating synthetic data...")
                return self.generate_realistic_crypto_data()
        else:
            print("No H5 file provided. Generating realistic synthetic data...")
            return self.generate_realistic_crypto_data()
    
    def load_h5_data(self):
        """Load OHLCV data from HDF5 file"""
        path = f"{self.symbol}/{self.timeframe}"
        data = pd.read_hdf(self.h5_path, path)
        
        # Standardize columns
        data.columns = [col.lower() for col in data.columns]
        
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ns')
            data.set_index('timestamp', inplace=True)
        
        # Use recent data for HFT
        data = data.tail(15000).copy()  # ~52 days of 5m data
        print(f"Loaded {len(data)} candles from H5")
        
        return data
    
    def generate_realistic_crypto_data(self, periods=15000):
        """Generate realistic crypto data with proper volatility clustering"""
        np.random.seed(42)
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=periods*5/1440)  # 5m periods
        timestamps = pd.date_range(start=start_date, periods=periods, freq='5min')
        
        # Base price for ZORA/USDT
        base_price = 0.085
        
        # Market regime simulation
        regime_length = 500  # Regime changes every ~41 hours
        n_regimes = periods // regime_length + 1
        
        returns = []
        volatilities = []
        
        for regime in range(n_regimes):
            start_idx = regime * regime_length
            end_idx = min((regime + 1) * regime_length, periods)
            regime_periods = end_idx - start_idx
            
            if regime_periods <= 0:
                break
            
            # Random regime type
            regime_type = np.random.choice(['bull', 'bear', 'sideways'], p=[0.3, 0.2, 0.5])
            
            if regime_type == 'bull':
                trend = np.random.uniform(0.0001, 0.0005)  # Positive trend
                base_vol = 0.003
            elif regime_type == 'bear':
                trend = np.random.uniform(-0.0005, -0.0001)  # Negative trend
                base_vol = 0.004  # Higher volatility in bear markets
            else:  # sideways
                trend = np.random.uniform(-0.0001, 0.0001)  # No trend
                base_vol = 0.002
            
            # Generate regime returns with volatility clustering
            regime_returns = []
            vol = base_vol
            
            for i in range(regime_periods):
                # GARCH-like volatility
                vol = 0.95 * vol + 0.05 * base_vol + 0.1 * abs(regime_returns[-1] if regime_returns else 0)
                vol = max(0.001, min(0.01, vol))  # Bounds
                
                # Return with trend and volatility
                ret = np.random.normal(trend, vol)
                regime_returns.append(ret)
                volatilities.append(vol)
            
            returns.extend(regime_returns)
        
        # Ensure we have exactly 'periods' returns
        returns = returns[:periods]
        volatilities = volatilities[:periods]
        
        # Generate price path
        log_prices = np.cumsum([np.log(base_price)] + returns)
        prices = np.exp(log_prices)
        
        # Generate realistic OHLC with proper alignment
        opens = prices[:-1]  # Remove last price for alignment (shape: periods)
        closes = prices[1:]  # Shift by 1 (shape: periods)
        
        # Generate highs and lows with realistic spreads - align with opens/closes
        spread_pct = np.array(volatilities) * 0.5  # Use full volatilities array
        
        highs = np.maximum(opens, closes) * (1 + spread_pct + np.abs(np.random.normal(0, spread_pct/2)))
        lows = np.minimum(opens, closes) * (1 - spread_pct - np.abs(np.random.normal(0, spread_pct/2)))
        
        # Generate volume with correlation to volatility
        base_volume = 1000000
        volume_mult = 1 + np.array(volatilities[1:]) * 10  # Higher vol = higher volume
        volumes = np.random.lognormal(np.log(base_volume), 0.5) * volume_mult
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=timestamps[1:])  # Align with closes
        
        print(f"Generated {len(df)} realistic crypto candles")
        return df
    
    def prepare_data(self):
        """Prepare data with market regime detection and splits"""
        # Market regime indicators
        self.data['sma_200'] = self.data['close'].rolling(200, min_periods=50).mean()
        self.data['bull_market'] = self.data['close'] > self.data['sma_200']
        
        # Volatility measure
        self.data['volatility'] = self.data['close'].pct_change().rolling(20).std()
        
        # Time-based features
        if hasattr(self.data.index, 'hour'):
            self.data['hour'] = self.data.index.hour
            # Active trading hours (higher volume/volatility)
            self.data['active_hours'] = self.data['hour'].isin([8, 9, 10, 14, 15, 16, 20, 21, 22])
        
        # Walk-forward splits for HFT
        self.create_walk_forward_splits()
        
    def create_walk_forward_splits(self):
        """Create walk-forward validation splits optimized for HFT"""
        n = len(self.data)
        
        # HFT needs more recent data, smaller windows
        train_size = 3000    # ~10 days for training
        val_size = 1000      # ~3.5 days for validation  
        test_size = 1000     # ~3.5 days for testing
        step_size = 500      # ~1.7 days step forward
        
        self.splits = []
        
        for start in range(0, n - train_size - val_size - test_size, step_size):
            train_end = start + train_size
            val_end = train_end + val_size
            test_end = val_end + test_size
            
            if test_end > n:
                break
                
            split = {
                'train': self.data.iloc[start:train_end].copy(),
                'val': self.data.iloc[train_end:val_end].copy(),
                'test': self.data.iloc[val_end:test_end].copy()
            }
            self.splits.append(split)
        
        print(f"Created {len(self.splits)} walk-forward splits")
        
        # Use the most recent split for optimization
        if self.splits:
            self.current_split = self.splits[-1]
            print(f"Using most recent split - Train: {len(self.current_split['train'])}, "
                  f"Val: {len(self.current_split['val'])}, Test: {len(self.current_split['test'])}")
    
    def calculate_indicators(self, data, rsi_length, mfi_length):
        """Calculate RSI and MFI indicators with proper error handling"""
        try:
            # RSI calculation
            delta = data['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Use EMA for faster response in HFT
            alpha = 2.0 / (rsi_length + 1)
            avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
            avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
            
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # MFI calculation
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            price_change = typical_price.diff()
            positive_flow = money_flow.where(price_change > 0, 0)
            negative_flow = money_flow.where(price_change <= 0, 0)
            
            # Use EMA for MFI too
            alpha_mfi = 2.0 / (mfi_length + 1)
            pos_mf_ema = positive_flow.ewm(alpha=alpha_mfi, adjust=False).mean()
            neg_mf_ema = negative_flow.ewm(alpha=alpha_mfi, adjust=False).mean()
            
            mf_ratio = pos_mf_ema / (neg_mf_ema + 1e-10)
            mfi = 100 - (100 / (1 + mf_ratio))
            
            return rsi.fillna(50), mfi.fillna(50)
            
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            n = len(data)
            return pd.Series([50] * n, index=data.index), pd.Series([50] * n, index=data.index)
    
    def advanced_backtest(self, params, data):
        """Advanced backtesting with realistic HFT constraints"""
        (rsi_length, mfi_length, oversold_level, overbought_level, 
         stop_loss_atr, take_profit_ratio, cooldown_periods, 
         trend_filter, regime_adaptive) = params
        
        # Calculate indicators
        rsi, mfi = self.calculate_indicators(data, int(rsi_length), int(mfi_length))
        
        # Calculate ATR for stops
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=14, adjust=False).mean()
        
        # Regime-adaptive thresholds
        if regime_adaptive:
            oversold_adj = np.where(data['bull_market'], 
                                   oversold_level + 5, oversold_level - 5)
            overbought_adj = np.where(data['bull_market'], 
                                     overbought_level + 5, overbought_level - 5)
        else:
            oversold_adj = oversold_level
            overbought_adj = overbought_level
        
        # Trading simulation
        position = 0
        entry_price = 0
        entry_time = None
        balance = self.initial_balance
        trades = []
        equity_curve = []
        last_trade_idx = -cooldown_periods
        
        max_position_time = 24  # Max 2 hours (24 * 5min) for HFT
        
        for i in range(max(int(rsi_length), int(mfi_length)) + 20, len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            current_atr = atr.iloc[i]
            
            # Calculate current equity
            if position == 0:
                current_equity = balance
            else:
                current_equity = position * current_price
            equity_curve.append(current_equity)
            
            # Position management
            if position > 0:
                # Time-based exit (HFT constraint)
                time_in_position = i - entry_idx
                if time_in_position >= max_position_time:
                    # Exit due to time limit
                    exit_price = current_price * (1 - self.slippage - self.taker_fee)
                    balance = position * exit_price
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'duration': time_in_position,
                        'exit_reason': 'time_limit',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    
                    position = 0
                    last_trade_idx = i
                    continue
                
                # ATR-based stop loss
                if current_atr > 0:
                    stop_price = entry_price - (current_atr * stop_loss_atr)
                    if current_price <= stop_price:
                        exit_price = stop_price * (1 - self.slippage - self.taker_fee)
                        balance = position * exit_price
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return': (exit_price / entry_price) - 1,
                            'duration': time_in_position,
                            'exit_reason': 'stop_loss',
                            'entry_time': entry_time,
                            'exit_time': current_time
                        })
                        
                        position = 0
                        last_trade_idx = i
                        continue
                
                # Take profit
                if current_price >= entry_price * (1 + take_profit_ratio):
                    exit_price = current_price * (1 - self.slippage - self.taker_fee)
                    balance = position * exit_price
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'duration': time_in_position,
                        'exit_reason': 'take_profit',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    
                    position = 0
                    last_trade_idx = i
                    continue
                
                # Signal-based exit
                if (rsi.iloc[i] > overbought_adj or mfi.iloc[i] > overbought_adj):
                    exit_price = current_price * (1 - self.slippage - self.taker_fee)
                    balance = position * exit_price
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'duration': time_in_position,
                        'exit_reason': 'signal_exit',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    
                    position = 0
                    last_trade_idx = i
                    continue
            
            # Entry logic
            elif position == 0 and (i - last_trade_idx) >= cooldown_periods:
                # Check for buy signals
                rsi_signal = rsi.iloc[i] < oversold_adj
                mfi_signal = mfi.iloc[i] < oversold_adj
                
                # Trend filter
                if trend_filter:
                    # Simple trend: price above/below short MA
                    ma_short = data['close'].iloc[i-10:i].mean()
                    trend_ok = current_price > ma_short
                else:
                    trend_ok = True
                
                # Combined entry condition
                if (rsi_signal or mfi_signal) and trend_ok:
                    entry_price = current_price * (1 + self.slippage + self.taker_fee)
                    position = (balance * 0.98) / entry_price  # Use 98% of balance
                    balance = 0
                    entry_time = current_time
                    entry_idx = i
                    last_trade_idx = i
        
        # Close any remaining position
        if position > 0:
            final_price = data['close'].iloc[-1] * (1 - self.slippage - self.taker_fee)
            balance = position * final_price
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': (final_price / entry_price) - 1,
                'duration': len(data) - entry_idx,
                'exit_reason': 'final_close',
                'entry_time': entry_time,
                'exit_time': data.index[-1]
            })
        
        # Calculate performance metrics
        return self.calculate_comprehensive_metrics(trades, equity_curve, data)
    
    def calculate_comprehensive_metrics(self, trades, equity_curve, data):
        """Calculate comprehensive performance metrics"""
        if len(trades) < 3:
            return {
                'score': -100,
                'total_return': -1,
                'sharpe_ratio': -10,
                'max_drawdown': -1,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': len(trades),
                'avg_duration': 0,
                'calmar_ratio': -10
            }
        
        # Basic metrics
        returns = [t['return'] for t in trades]
        total_return = np.prod([1 + r for r in returns]) - 1
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        avg_duration = np.mean([t['duration'] for t in trades])
        
        # Profit factor
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0
        
        # Risk metrics
        if len(equity_curve) > 1:
            equity_series = pd.Series(equity_curve)
            drawdowns = (equity_series / equity_series.expanding().max()) - 1
            max_drawdown = drawdowns.min()
            
            # Sharpe ratio from equity curve
            equity_returns = equity_series.pct_change().dropna()
            if len(equity_returns) > 0 and equity_returns.std() > 0:
                sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252 * 288)
            else:
                sharpe_ratio = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else total_return
        
        # Advanced scoring for HFT
        frequency_score = min(len(trades) / (len(data) / 100), 1.0)  # Normalize frequency
        consistency_score = 1 - np.std(returns) if returns else 0
        
        # Multi-objective score
        score = (
            total_return * 25 +           # 25% weight on returns
            sharpe_ratio * 0.3 +          # 30% weight on risk-adjusted returns  
            win_rate * 20 +               # 20% weight on win rate
            calmar_ratio * 0.15 +         # 15% weight on drawdown-adjusted returns
            frequency_score * 5 +         # 5% weight on trade frequency
            consistency_score * 5         # 5% weight on consistency
        )
        
        return {
            'score': score,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades),
            'avg_duration': avg_duration,
            'calmar_ratio': calmar_ratio,
            'consistency': consistency_score
        }
    
    def objective_function(self, params):
        """Multi-period objective function with robustness checks"""
        self.call_count += 1
        
        try:
            # Test on current split
            train_result = self.advanced_backtest(params, self.current_split['train'])
            val_result = self.advanced_backtest(params, self.current_split['val'])
            
            # Robustness test on multiple splits if available
            if len(self.splits) > 3:
                # Test on 3 random older splits
                robustness_scores = []
                test_splits = np.random.choice(self.splits[:-1], size=min(3, len(self.splits)-1), replace=False)
                
                for split in test_splits:
                    rob_result = self.advanced_backtest(params, split['val'])
                    if rob_result['num_trades'] >= 3:
                        robustness_scores.append(rob_result['score'])
                
                robustness_penalty = 1.0 if not robustness_scores else np.std(robustness_scores) * 0.1
            else:
                robustness_penalty = 0
            
            # Combined score with robustness penalty
            combined_score = (
                0.3 * train_result['score'] + 
                0.7 * val_result['score'] - 
                robustness_penalty
            )
            
            # Store best results
            self.best_results.append({
                'params': params,
                'train_score': train_result['score'],
                'val_score': val_result['score'],
                'combined_score': combined_score,
                'train_metrics': train_result,
                'val_metrics': val_result
            })
            
            # Progress reporting
            if self.call_count % 10 == 0:
                best_so_far = max(self.best_results, key=lambda x: x['combined_score'])
                print(f"Call {self.call_count}: Current={combined_score:.2f}, "
                      f"Best={best_so_far['combined_score']:.2f}, "
                      f"Trades={val_result['num_trades']}")
            
            return -combined_score  # Minimize negative score
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000  # High penalty for errors
    
    def optimize_hft_parameters(self, n_calls=100):
        """Advanced HFT parameter optimization with proper bounds"""
        
        # HFT-optimized parameter space
        space = [
            Integer(3, 10, name='rsi_length'),           # Very short for HFT
            Integer(3, 10, name='mfi_length'),           # Very short for HFT
            Integer(15, 35, name='oversold_level'),      # Sensitive levels
            Integer(65, 85, name='overbought_level'),    # Sensitive levels
            Real(0.8, 2.5, name='stop_loss_atr'),        # ATR-based stops
            Real(0.01, 0.05, name='take_profit_ratio'),  # 1-5% take profit
            Integer(1, 5, name='cooldown_periods'),      # Signal cooldown
            Categorical([True, False], name='trend_filter'),    # Trend filter
            Categorical([True, False], name='regime_adaptive')  # Adaptive thresholds
        ]
        
        print(f"\n🚀 Starting Advanced HFT Optimization for {self.symbol} {self.timeframe}")
        print(f"📊 Data: {len(self.data)} candles, {len(self.splits)} walk-forward splits")
        print(f"🎯 Optimization calls: {n_calls}")
        print(f"💰 Fee structure: {self.taker_fee*100:.1f}% + {self.slippage*100:.2f}% slippage")
        print("="*60)
        
        self.call_count = 0
        self.best_results = []
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=min(20, n_calls//3),
            random_state=42,
            acq_func='EI',  # Expected Improvement
            n_jobs=1
        )
        
        best_params = result.x
        
        print(f"\n✅ Optimization Complete! Total calls: {self.call_count}")
        print("="*60)
        print("🏆 BEST PARAMETERS FOUND:")
        print(f"   RSI Length: {best_params[0]}")
        print(f"   MFI Length: {best_params[1]}")
        print(f"   Oversold Level: {best_params[2]}")
        print(f"   Overbought Level: {best_params[3]}")
        print(f"   Stop Loss ATR: {best_params[4]:.2f}")
        print(f"   Take Profit: {best_params[5]*100:.1f}%")
        print(f"   Cooldown Periods: {best_params[6]}")
        print(f"   Trend Filter: {best_params[7]}")
        print(f"   Regime Adaptive: {best_params[8]}")
        
        return best_params
    
    def comprehensive_evaluation(self, params):
        """Comprehensive evaluation across all time periods"""
        print(f"\n📈 COMPREHENSIVE EVALUATION")
        print("="*60)
        
        all_results = []
        
        # Evaluate on all splits
        for i, split in enumerate(self.splits):
            train_result = self.advanced_backtest(params, split['train'])
            val_result = self.advanced_backtest(params, split['val'])
            test_result = self.advanced_backtest(params, split['test'])
            
            all_results.append({
                'period': i + 1,
                'train': train_result,
                'val': val_result,
                'test': test_result
            })
        
        # Aggregate statistics
        train_returns = [r['train']['total_return'] for r in all_results]
        val_returns = [r['val']['total_return'] for r in all_results]
        test_returns = [r['test']['total_return'] for r in all_results]
        
        train_sharpe = [r['train']['sharpe_ratio'] for r in all_results]
        val_sharpe = [r['val']['sharpe_ratio'] for r in all_results]
        test_sharpe = [r['test']['sharpe_ratio'] for r in all_results]
        
        # Print summary
        print(f"{'Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
        print("-" * 65)
        print(f"{'Avg Return':<20} {np.mean(train_returns):<15.2%} "
              f"{np.mean(val_returns):<15.2%} {np.mean(test_returns):<15.2%}")
        print(f"{'Std Return':<20} {np.std(train_returns):<15.2%} "
              f"{np.std(val_returns):<15.2%} {np.std(test_returns):<15.2%}")
        print(f"{'Avg Sharpe':<20} {np.mean(train_sharpe):<15.2f} "
              f"{np.mean(val_sharpe):<15.2f} {np.mean(test_sharpe):<15.2f}")
        print(f"{'Win Rate Avg':<20} "
              f"{np.mean([r['train']['win_rate'] for r in all_results]):<15.2%} "
              f"{np.mean([r['val']['win_rate'] for r in all_results]):<15.2%} "
              f"{np.mean([r['test']['win_rate'] for r in all_results]):<15.2%}")
        
        # Consistency analysis
        positive_periods = sum(1 for r in test_returns if r > 0)
        consistency = positive_periods / len(test_returns)
        
        print(f"\n📊 CONSISTENCY ANALYSIS:")
        print(f"   Profitable periods: {positive_periods}/{len(test_returns)} ({consistency:.1%})")
        print(f"   Best period return: {max(test_returns):.2%}")
        print(f"   Worst period return: {min(test_returns):.2%}")
        
        return all_results
    
    def generate_ensemble_parameters(self, n_top=5):
        """Generate ensemble of top parameter sets"""
        if len(self.best_results) < n_top:
            print(f"Only {len(self.best_results)} results available for ensemble")
            n_top = len(self.best_results)
        
        # Sort by combined score
        sorted_results = sorted(self.best_results, 
                               key=lambda x: x['combined_score'], reverse=True)
        
        ensemble_params = []
        for i in range(n_top):
            result = sorted_results[i]
            params = result['params']
            
            param_dict = {
                'rsi_length': int(params[0]),
                'mfi_length': int(params[1]),
                'oversold_level': int(params[2]),
                'overbought_level': int(params[3]),
                'stop_loss_atr': float(params[4]),
                'take_profit_ratio': float(params[5]),
                'cooldown_periods': int(params[6]),
                'trend_filter': bool(params[7]),
                'regime_adaptive': bool(params[8]),
                'score': result['combined_score'],
                'val_return': result['val_metrics']['total_return'],
                'val_sharpe': result['val_metrics']['sharpe_ratio']
            }
            ensemble_params.append(param_dict)
        
        return ensemble_params
    
    def save_optimization_results(self, best_params, all_results, ensemble_params):
        """Save comprehensive optimization results"""
        
        # Convert best params to dict
        best_param_dict = {
            'rsi_length': int(best_params[0]),
            'mfi_length': int(best_params[1]),
            'oversold_level': int(best_params[2]),
            'overbought_level': int(best_params[3]),
            'stop_loss_atr': float(best_params[4]),
            'take_profit_ratio': float(best_params[5]),
            'cooldown_periods': int(best_params[6]),
            'trend_filter': bool(best_params[7]),
            'regime_adaptive': bool(best_params[8])
        }
        
        # Create results summary
        output = {
            'optimization_info': {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'data_points': len(self.data),
                'optimization_calls': self.call_count,
                'walk_forward_splits': len(self.splits)
            },
            'best_parameters': best_param_dict,
            'ensemble_parameters': ensemble_params,
            'performance_summary': {
                'train_avg_return': np.mean([r['train']['total_return'] for r in all_results]),
                'val_avg_return': np.mean([r['val']['total_return'] for r in all_results]),
                'test_avg_return': np.mean([r['test']['total_return'] for r in all_results]),
                'consistency': sum(1 for r in all_results if r['test']['total_return'] > 0) / len(all_results)
            },
            'hft_features': [
                'EMA-based indicators for faster response',
                'ATR-based dynamic stops',
                'Time-limited positions (2 hours max)',
                'Realistic trading fees and slippage',
                'Regime-adaptive thresholds',
                'Walk-forward validation',
                'Multi-objective optimization'
            ]
        }
        
        # Save main results
        filename = f"hft_optimized_{self.symbol}_{self.timeframe}_{output['optimization_info']['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save best params in strategy format
        strategy_params = {
            'rsi_length': best_param_dict['rsi_length'],
            'mfi_length': best_param_dict['mfi_length'],
            'oversold_level': best_param_dict['oversold_level'],
            'overbought_level': best_param_dict['overbought_level'],
            'require_trend': best_param_dict['trend_filter'],
            'atr_multiplier': best_param_dict['stop_loss_atr'],
            'signal_cooldown': best_param_dict['cooldown_periods']
        }
        
        with open('optimized_params.json', 'w') as f:
            json.dump(strategy_params, f, indent=2)
        
        with open('ensemble_params.json', 'w') as f:
            json.dump([{k: v for k, v in p.items() 
                       if k in strategy_params.keys()} for p in ensemble_params], f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Full results: {filename}")
        print(f"   ⚙️  Strategy params: optimized_params.json")
        print(f"   🎯 Ensemble params: ensemble_params.json")
        
        return filename


def run_advanced_hft_optimization(h5_path=None, symbol="ZORAUSDT", timeframe="5m", n_calls=150):
    """Main function to run the advanced HFT optimization"""
    
    print("🚀 ADVANCED CRYPTO HFT OPTIMIZER")
    print("="*50)
    
    # Initialize optimizer
    optimizer = AdvancedCryptoHFTOptimizer(
        h5_path=h5_path,
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=10000
    )
    
    # Run optimization
    best_params = optimizer.optimize_hft_parameters(n_calls=n_calls)
    
    # Comprehensive evaluation
    all_results = optimizer.comprehensive_evaluation(best_params)
    
    # Generate ensemble
    ensemble_params = optimizer.generate_ensemble_parameters(n_top=5)
    
    # Save results
    results_file = optimizer.save_optimization_results(
        best_params, all_results, ensemble_params
    )
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"✅ Best strategy ready for deployment")
    print(f"📊 Ensemble of 5 strategies available for diversification")
    
    return optimizer, best_params, all_results, ensemble_params


# Example usage
if __name__ == "__main__":
    # Run for ZORA/USDT 5m with synthetic data
    optimizer, best_params, results, ensemble = run_advanced_hft_optimization(
        h5_path=None,  # Will use synthetic data
        symbol="ZORAUSDT",
        timeframe="5m",
        n_calls=100  # Reduce for testing
    )