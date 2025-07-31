#!/usr/bin/env python3
"""
ETH/USDT Futures Backtest - FIXED VERSION
Optimized RSI/MFI Strategy with Proper Parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import time
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EthBacktesterFixed:
    def __init__(self, initial_balance: float = 10000):
        # Backtest parameters
        self.timeframe = "1m"
        self.start_date = "2025-04-01"
        self.end_date = "2025-07-01"
        self.symbol = "ETHUSDT"
        self.initial_balance = initial_balance
        
        # FIXED Strategy parameters (based on research)
        self.rsi_length = 14          # Standard RSI period
        self.mfi_length = 14          # Standard MFI period
        self.oversold_level = 20      # was 25 - more selective
        self.overbought_level = 80    # was 75 - more selective  
        self.signal_cooldown = 60     # was 30 - reduce whipsaws
        self.trend_filter_period = 20 # was 50 - more responsive
        self.structure_buffer_pct = 0.5  # Larger buffer to avoid whipsaws
        
        # FIXED Risk management
        self.fixed_risk_usd = 100.0
        self.stop_loss_pct = 0.025    # 2.5% stop loss (wider)
        self.take_profit_pct = 0.075  # 7.5% take profit (3:1 RR)
        self.trailing_stop_pct = 0.015 # 1.5% trailing
        self.profit_lock_threshold = 2.0  # 2% to activate trailing
        
        # FIXED Signal management
        self.signal_cooldown = 30     # 30 minutes between signals
        self.require_both_indicators = True  # Both RSI AND MFI must agree
        self.enable_position_flipping = False # Disable aggressive flipping
        self.trend_filter_period = 50 # 50-period EMA for trend filter
        
        # State tracking
        self.data = None
        self.trades = []
        self.balance = initial_balance
        self.position = None
        self.last_signal = None
        self.last_signal_time = None
        self.profit_lock_active = False
        
    def fetch_binance_data(self) -> pd.DataFrame:
        """Fetch real Binance futures data for ETH/USDT"""
        print(f"📊 Fetching {self.symbol} data from Binance...")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        
        start_ms = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ms = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_start = start_ms
        
        while current_start < end_ms:
            try:
                url = "https://fapi.binance.com/fapi/v1/klines"
                params = {
                    'symbol': self.symbol,
                    'interval': self.timeframe,
                    'startTime': current_start,
                    'endTime': min(current_start + (1500 * 60 * 1000), end_ms),
                    'limit': 1500
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                klines = response.json()
                if not klines:
                    break
                
                for kline in klines:
                    all_data.append({
                        'timestamp': pd.to_datetime(kline[0], unit='ms'),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                current_start = klines[-1][0] + 60000
                
                # Less verbose progress
                if len(all_data) % 15000 == 0:
                    print(f"✅ Fetched {len(all_data)} candles...")
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {e}")
                break
        
        if not all_data:
            print("❌ No data fetched. Using synthetic data for demo...")
            return self._generate_synthetic_data()
        
        df = pd.DataFrame(all_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"✅ Data loaded: {len(df)} candles")
        print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate realistic synthetic data if Binance API fails"""
        print("🔄 Generating synthetic ETH/USDT data...")
        
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='1min')
        
        np.random.seed(42)
        data = []
        current_price = 3200 + np.random.random() * 400
        
        for i, ts in enumerate(timestamps[:50000]):
            volatility = 0.0008 + np.random.random() * 0.0012  # Reduced volatility
            trend = np.sin(i / 1440) * 0.0003  # Smaller trend component
            noise = (np.random.random() - 0.5) * volatility
            
            price_change = (trend + noise) * current_price
            current_price = max(current_price + price_change, 100)
            
            open_price = current_price if i == 0 else data[-1]['close']
            close_price = current_price
            high_price = max(open_price, close_price) * (1 + np.random.random() * 0.001)
            low_price = min(open_price, close_price) * (1 - np.random.random() * 0.001)
            volume = 1000 + np.random.random() * 2000
            
            data.append({
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Synthetic data generated: {len(df)} candles")
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_length
        
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Use SMA instead of EMA for more stable RSI
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).clip(0, 100)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, volume: pd.Series, period: int = None) -> pd.Series:
        """Calculate Money Flow Index"""
        if period is None:
            period = self.mfi_length
        
        if len(close) < period + 1:
            return pd.Series([50] * len(close), index=close.index)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        mf_sign = typical_price.diff()
        positive_mf = money_flow.where(mf_sign > 0, 0)
        negative_mf = money_flow.where(mf_sign <= 0, 0)
        
        # Use SMA for more stable MFI
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mf_ratio = positive_mf_sum / negative_mf_sum.replace(0, 0.0001)
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi.fillna(50).clip(0, 100)
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def get_structure_stop(self, df: pd.DataFrame, action: str, 
                          entry_price: float, current_idx: int) -> float:
        """Calculate structure-based stop loss with wider parameters"""
        lookback = min(self.structure_lookback, current_idx)
        
        if lookback < 10:
            # Fallback to fixed percentage
            buffer = entry_price * (self.structure_buffer_pct / 100)
            if action == 'SELL':
                return entry_price * (1 + self.stop_loss_pct) + buffer
            else:
                return entry_price * (1 - self.stop_loss_pct) - buffer
        
        start_idx = max(0, current_idx - lookback)
        recent_data = df.iloc[start_idx:current_idx + 1]
        buffer = entry_price * (self.structure_buffer_pct / 100)
        
        if action == 'SELL':  # Short position
            swing_high = recent_data['high'].max()
            return swing_high + buffer
        else:  # Long position
            swing_low = recent_data['low'].min()
            return swing_low - buffer
    
    def calculate_position_size(self, entry_price: float, structure_stop: float) -> float:
        """Calculate position size based on fixed risk"""
        stop_distance = abs(entry_price - structure_stop)
        position_size = self.fixed_risk_usd / stop_distance
        
        # Don't risk more than 5% of balance
        position_value = position_size * entry_price
        max_position_value = self.balance * 0.05
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size
    
    def get_take_profit_price(self, entry_price: float, side: str, 
                             structure_stop: float) -> float:
        """Calculate take profit price (3:1 R:R)"""
        risk_distance = abs(entry_price - structure_stop)
        reward_distance = risk_distance * 3  # 3:1 reward ratio
        
        if side == 'BUY':
            return entry_price + reward_distance
        else:
            return entry_price - reward_distance
    
    def is_trending_up(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if market is trending up using EMA"""
        if 'ema' not in df.columns or idx < self.trend_filter_period:
            return True  # Default to allow trades
        
        current_price = df['close'].iloc[idx]
        ema_value = df['ema'].iloc[idx]
        return current_price > ema_value
    
    def detect_signal(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Detect RSI/MFI trading signals with improved logic"""
        if idx < max(self.rsi_length, self.mfi_length, self.trend_filter_period) + 5:
            return None
        
        current_data = df.iloc[idx]
        current_rsi = df['rsi'].iloc[idx]
        current_mfi = df['mfi'].iloc[idx]
        
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None
        
        # Check signal cooldown
        if self.last_signal_time:
            time_diff = (current_data.name - self.last_signal_time).total_seconds() / 60
            if time_diff < self.signal_cooldown:
                return None
        
        # Trend filter
        is_uptrend = self.is_trending_up(df, idx)
        
        # BUY signal - Both RSI and MFI oversold AND uptrend
        if self.require_both_indicators:
            buy_condition = (current_rsi < self.oversold_level and 
                           current_mfi < self.oversold_level and 
                           is_uptrend and 
                           self.last_signal != 'BUY')
        else:
            buy_condition = ((current_rsi < self.oversold_level or 
                            current_mfi < self.oversold_level) and 
                           is_uptrend and 
                           self.last_signal != 'BUY')
        
        if buy_condition:
            self.last_signal = 'BUY'
            self.last_signal_time = current_data.name
            structure_stop = self.get_structure_stop(df, 'BUY', current_data['close'], idx)
            
            return {
                'action': 'BUY',
                'price': current_data['close'],
                'rsi': current_rsi,
                'mfi': current_mfi,
                'timestamp': current_data.name,
                'structure_stop': structure_stop,
                'signal_type': 'RSI_MFI'
            }
        
        # SELL signal - Both RSI and MFI overbought AND downtrend
        if self.require_both_indicators:
            sell_condition = (current_rsi > self.overbought_level and 
                            current_mfi > self.overbought_level and 
                            not is_uptrend and 
                            self.last_signal != 'SELL')
        else:
            sell_condition = ((current_rsi > self.overbought_level or 
                             current_mfi > self.overbought_level) and 
                            not is_uptrend and 
                            self.last_signal != 'SELL')
        
        if sell_condition:
            self.last_signal = 'SELL'
            self.last_signal_time = current_data.name
            structure_stop = self.get_structure_stop(df, 'SELL', current_data['close'], idx)
            
            return {
                'action': 'SELL',
                'price': current_data['close'],
                'rsi': current_rsi,
                'mfi': current_mfi,
                'timestamp': current_data.name,
                'structure_stop': structure_stop,
                'signal_type': 'RSI_MFI'
            }
        
        return None
    
    def open_position(self, signal: Dict, timestamp: pd.Timestamp) -> bool:
        """Open a new position"""
        # Close existing position if opposite signal
        if self.position and self.position['side'] != signal['action']:
            self.close_position(timestamp, "Opposite Signal")
        
        if self.position:  # Still have position after attempted close
            return False
        
        entry_price = signal['price']
        structure_stop = signal['structure_stop']
        position_size = self.calculate_position_size(entry_price, structure_stop)
        
        # Calculate risk/reward
        side = signal['action']
        take_profit = self.get_take_profit_price(entry_price, side, structure_stop)
        
        # Calculate actual amounts
        if side == 'BUY':
            risk_amount = position_size * (entry_price - structure_stop)
            reward_amount = position_size * (take_profit - entry_price)
        else:
            risk_amount = position_size * (structure_stop - entry_price)
            reward_amount = position_size * (entry_price - take_profit)
        
        self.position = {
            'side': side,
            'entry_price': entry_price,
            'size': position_size,
            'structure_stop': structure_stop,
            'take_profit': take_profit,
            'open_time': timestamp,
            'risk_amount': abs(risk_amount),
            'reward_amount': abs(reward_amount),
            'signal_type': signal['signal_type']
        }
        
        self.profit_lock_active = False
        return True
    
    def close_position(self, timestamp: pd.Timestamp, reason: str, 
                      close_price: float = None) -> bool:
        """Close current position"""
        if not self.position:
            return False
        
        if close_price is None:
            close_price = self.position['entry_price']
        
        # Calculate P&L
        if self.position['side'] == 'BUY':
            pnl = (close_price - self.position['entry_price']) * self.position['size']
        else:
            pnl = (self.position['entry_price'] - close_price) * self.position['size']
        
        # Calculate percentage return
        pnl_pct = (pnl / (self.position['entry_price'] * self.position['size'])) * 100
        
        # Calculate duration
        duration = timestamp - self.position['open_time']
        
        # Record trade
        trade = {
            'open_time': self.position['open_time'],
            'close_time': timestamp,
            'duration_minutes': duration.total_seconds() / 60,
            'side': self.position['side'],
            'entry_price': self.position['entry_price'],
            'close_price': close_price,
            'size': self.position['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'risk_amount': self.position['risk_amount'],
            'reward_amount': self.position['reward_amount'],
            'reason': reason,
            'signal_type': self.position['signal_type']
        }
        
        self.trades.append(trade)
        self.balance += pnl
        self.position = None
        self.profit_lock_active = False
        
        return True
    
    def check_position_management(self, current_price: float, timestamp: pd.Timestamp) -> bool:
        """Check stop loss, take profit, and profit lock conditions"""
        if not self.position:
            return False
        
        # Check stop loss
        if self.position['side'] == 'BUY':
            if current_price <= self.position['structure_stop']:
                self.close_position(timestamp, "Stop Loss", current_price)
                return True
            
            # Check take profit
            if current_price >= self.position['take_profit']:
                self.close_position(timestamp, "Take Profit", current_price)
                return True
            
            # Check profit lock activation
            profit_pct = ((current_price - self.position['entry_price']) / self.position['entry_price']) * 100
            
        else:  # SELL position
            if current_price >= self.position['structure_stop']:
                self.close_position(timestamp, "Stop Loss", current_price)
                return True
            
            # Check take profit
            if current_price <= self.position['take_profit']:
                self.close_position(timestamp, "Take Profit", current_price)
                return True
            
            # Check profit lock activation
            profit_pct = ((self.position['entry_price'] - current_price) / self.position['entry_price']) * 100
        
        # Activate profit lock if not already active
        if not self.profit_lock_active and profit_pct >= self.profit_lock_threshold:
            self.profit_lock_active = True
            
            # Update stop to trailing stop
            trailing_distance = current_price * self.trailing_stop_pct
            if self.position['side'] == 'BUY':
                self.position['structure_stop'] = max(self.position['structure_stop'], 
                                                    current_price - trailing_distance)
            else:
                self.position['structure_stop'] = min(self.position['structure_stop'], 
                                                    current_price + trailing_distance)
        
        return False
    
    def run_backtest(self) -> Dict:
        """Run the complete backtest"""
        print("🚀 Starting FIXED ETH/USDT Futures Backtest...")
        print("=" * 60)
        print("🔧 OPTIMIZED PARAMETERS:")
        print(f"   RSI/MFI Levels: {self.oversold_level}/{self.overbought_level} (was 40/50)")
        print(f"   Signal Cooldown: {self.signal_cooldown} minutes")
        print(f"   Trend Filter: {self.trend_filter_period}-period EMA")
        print(f"   Both Indicators Required: {self.require_both_indicators}")
        print(f"   Position Flipping: {self.enable_position_flipping}")
        print("=" * 60)
        
        # Fetch data
        self.data = self.fetch_binance_data()
        
        if self.data is None or len(self.data) < 100:
            print("❌ Insufficient data for backtest")
            return None
        
        # Calculate indicators
        print("📊 Calculating indicators...")
        self.data['rsi'] = self.calculate_rsi(self.data['close'])
        self.data['mfi'] = self.calculate_mfi(
            self.data['high'], self.data['low'], 
            self.data['close'], self.data['volume']
        )
        self.data['ema'] = self.calculate_ema(self.data['close'], self.trend_filter_period)
        
        # Reset state
        self.trades = []
        self.balance = self.initial_balance
        self.position = None
        self.last_signal = None
        self.last_signal_time = None
        self.profit_lock_active = False
        
        print("⚡ Running FIXED backtest simulation...")
        
        # Main backtest loop
        total_bars = len(self.data)
        for idx in range(total_bars):
            current_data = self.data.iloc[idx]
            current_price = current_data['close']
            timestamp = current_data.name
            
            # Progress indicator (less frequent)
            if idx % 20000 == 0:
                progress = (idx / total_bars) * 100
                print(f"📈 Progress: {progress:.1f}% - Price: ${current_price:.2f} - Trades: {len(self.trades)}")
            
            # Check position management first
            if self.check_position_management(current_price, timestamp):
                continue
            
            # Check for new signals (no structure break flipping in fixed version)
            signal = self.detect_signal(self.data, idx)
            if signal:
                self.open_position(signal, timestamp)
        
        # Close any remaining position
        if self.position:
            final_price = self.data['close'].iloc[-1]
            final_timestamp = self.data.index[-1]
            self.close_position(final_timestamp, "Backtest End", final_price)
        
        # Calculate results
        results = self.calculate_results()
        self.display_results(results)
        
        return results
    
    def calculate_results(self) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_balance) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        
        # Risk metrics
        max_drawdown = 0
        peak_balance = self.initial_balance
        running_balance = self.initial_balance
        
        for _, trade in trades_df.iterrows():
            running_balance += trade['pnl']
            if running_balance > peak_balance:
                peak_balance = running_balance
            drawdown = (peak_balance - running_balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Time metrics
        avg_duration = trades_df['duration_minutes'].mean()
        total_days = (self.data.index[-1] - self.data.index[0]).days
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'final_balance': self.balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_duration_minutes': avg_duration,
            'sharpe_ratio': sharpe_ratio,
            'total_days': total_days,
            'trades_per_day': total_trades / total_days if total_days > 0 else 0,
            'trades_df': trades_df
        }
    
    def display_results(self, results: Dict):
        """Display backtest results"""
        print("\n" + "=" * 60)
        print("📊 FIXED BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"📈 Strategy: FIXED RSI({self.rsi_length})/MFI({self.mfi_length}) + Structure Stops")
        print(f"🔧 Levels: {self.oversold_level}/{self.overbought_level} (was 40/50)")
        print(f"🎯 Period: {self.start_date} to {self.end_date} ({results['total_days']} days)")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance: ${results['final_balance']:,.2f}")
        print(f"📊 Total Return: {results['total_return_pct']:+.2f}%")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        
        print(f"\n🔢 TRADE STATISTICS")
        print(f"🎯 Total Trades: {results['total_trades']}")
        print(f"✅ Winning Trades: {results['winning_trades']}")
        print(f"❌ Losing Trades: {results['losing_trades']}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"⚡ Trades/Day: {results['trades_per_day']:.1f}")
        print(f"⏱️ Avg Duration: {results['avg_duration_minutes']:.0f} minutes")
        
        print(f"\n💵 P&L ANALYSIS")
        print(f"🟢 Average Win: ${results['avg_win']:,.2f}")
        print(f"🔴 Average Loss: ${results['avg_loss']:,.2f}")
        print(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
        print(f"📈 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        print(f"\n🎯 OPTIMIZATIONS APPLIED")
        print(f"🔧 RSI/MFI Levels: {self.oversold_level}/{self.overbought_level} (more extreme)")
        print(f"⏰ Signal Cooldown: {self.signal_cooldown} minutes")
        print(f"📈 Trend Filter: {self.trend_filter_period}-period EMA")
        print(f"🎯 Both Indicators Required: ✅")
        print(f"🔄 Position Flipping: ❌ (disabled)")
        
        # Show sample trades
        if not results['trades_df'].empty:
            print(f"\n📋 RECENT TRADES (Last 5)")
            recent_trades = results['trades_df'].tail().copy()
            recent_trades['duration_str'] = recent_trades['duration_minutes'].apply(lambda x: f"{int(x)}m")
            recent_trades['pnl_str'] = recent_trades['pnl'].apply(lambda x: f"${x:+.2f}")
            
            for _, trade in recent_trades.iterrows():
                side_emoji = "📈" if trade['side'] == 'BUY' else "📉"
                result_emoji = "✅" if trade['pnl'] > 0 else "❌"
                print(f"{side_emoji} {trade['side']} ${trade['entry_price']:.2f} → ${trade['close_price']:.2f} "
                      f"| {trade['pnl_str']} | {trade['duration_str']} | {trade['reason']} {result_emoji}")
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        if not results or 'trades_df' not in results:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('ETH/USDT FIXED Strategy Results', fontsize=16, fontweight='bold')
        
        # 1. Price chart with signals
        ax1 = axes[0, 0]
        price_data = self.data['close'].resample('1H').last()
        ax1.plot(price_data.index, price_data.values, label='ETH Price', alpha=0.7)
        
        # Mark trade entries/exits
        trades_df = results['trades_df']
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['side'] == 'BUY' else 'red'
            marker = '^' if trade['side'] == 'BUY' else 'v'
            ax1.scatter(trade['open_time'], trade['entry_price'], 
                       color=color, marker=marker, s=50, alpha=0.8)
        
        ax1.set_title('Price Chart with Trade Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax2 = axes[0, 1]
        equity_curve = [self.initial_balance]
        for _, trade in trades_df.iterrows():
            equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        trade_times = [self.data.index[0]] + trades_df['close_time'].tolist()
        ax2.plot(trade_times, equity_curve, color='blue', linewidth=2)
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Balance ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L distribution
        ax3 = axes[1, 0]
        ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss analysis
        ax4 = axes[1, 1]
        wins_losses = ['Wins', 'Losses']
        counts = [results['winning_trades'], results['losing_trades']]
        colors = ['green', 'red']
        ax4.pie(counts, labels=wins_losses, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f"Win Rate: {results['win_rate']:.1f}%")
        
        # 5. Trade duration analysis
        ax5 = axes[2, 0]
        ax5.hist(trades_df['duration_minutes'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax5.set_title('Trade Duration Distribution')
        ax5.set_xlabel('Duration (minutes)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Monthly returns
        ax6 = axes[2, 1]
        trades_df['month'] = trades_df['close_time'].dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        
        if len(monthly_pnl) > 0:
            bars = ax6.bar(range(len(monthly_pnl)), monthly_pnl.values, 
                          color=['green' if x > 0 else 'red' for x in monthly_pnl.values],
                          alpha=0.7)
            ax6.set_title('Monthly P&L')
            ax6.set_ylabel('P&L ($)')
            ax6.set_xticks(range(len(monthly_pnl)))
            ax6.set_xticklabels([str(x) for x in monthly_pnl.index], rotation=45)
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results: Dict, filename: str = None):
        """Export results to CSV"""
        if not results or 'trades_df' not in results:
            return
        
        if filename is None:
            filename = f"eth_backtest_FIXED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades_df = results['trades_df'].copy()
        trades_df.to_csv(filename, index=False)
        print(f"📁 Results exported to: {filename}")


def main():
    """Main execution function"""
    print("🚀 ETH/USDT Futures Backtester - FIXED VERSION")
    print("📊 Optimized RSI/MFI Strategy with Proper Parameters")
    print("=" * 60)
    
    # Initialize backtester
    backtester = EthBacktesterFixed(initial_balance=10000)
    
    try:
        # Run backtest
        results = backtester.run_backtest()
        
        if results and 'error' not in results:
            # Plot results
            backtester.plot_results(results)
            
            # Export results
            backtester.export_results(results)
            
            print("\n✅ FIXED Backtest completed successfully!")
            print(f"💰 Final P&L: ${results['total_pnl']:+,.2f} ({results['total_return_pct']:+.2f}%)")
            print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
            print(f"⚡ Trades/Day: {results['trades_per_day']:.1f}")
            
        else:
            print("❌ Backtest failed or no trades executed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrupted by user")
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()