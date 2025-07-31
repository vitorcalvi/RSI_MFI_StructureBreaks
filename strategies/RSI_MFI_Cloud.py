import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RSIMFICloudStrategy:
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self._load_config()
        self.last_signal = None
        
        # Break & Retest state management
        self.recent_breaks = []  # Track recent structure breaks
        self.retest_monitoring = None  # Current retest being monitored
        self.max_break_history = 3  # Keep last 3 breaks
        
        print("✅ Break & Retest pattern detection enabled")
    
    def _load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
        with open(params_file, 'r') as f:
            self.params = json.load(f)
    
    @property
    def symbol(self):
        return self.risk_manager.symbol
    
    def calculate_rsi(self, prices):
        period = self.params['rsi_length']
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
        
        prices = pd.Series(prices).astype(float)
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.ewm(span=period).mean()
        avg_loss = losses.ewm(span=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).clip(0, 100)

    def calculate_mfi(self, high, low, close, volume):
        period = self.params['mfi_length']
        if len(close) < period + 1:
            return pd.Series([50] * len(close), index=close.index)
        
        high = pd.Series(high).astype(float)
        low = pd.Series(low).astype(float)
        close = pd.Series(close).astype(float)
        volume = pd.Series(volume).astype(float)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        mf_sign = typical_price.diff()
        positive_mf = money_flow.where(mf_sign > 0, 0)
        negative_mf = money_flow.where(mf_sign <= 0, 0)
        
        positive_mf_ema = positive_mf.ewm(span=period).mean()
        negative_mf_ema = negative_mf.ewm(span=period).mean()
        
        mf_ratio = positive_mf_ema / negative_mf_ema.replace(0, 0.0001)
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi.fillna(50).clip(0, 100)
    
    def get_structure_stop(self, df, action, entry_price):
        """Calculate structure-based stop loss"""
        lookback = 20
        
        if len(df) < lookback:
            buffer = entry_price * 0.002
            if action == 'SELL':
                fixed_stop = entry_price * 1.015 + buffer
                print(f"📊 Using Fixed Stop (insufficient data): ${fixed_stop:.2f}")
                return fixed_stop
            else:
                fixed_stop = entry_price * 0.985 - buffer
                print(f"📊 Using Fixed Stop (insufficient data): ${fixed_stop:.2f}")
                return fixed_stop
        
        recent_data = df.tail(lookback)
        buffer = entry_price * 0.002
        
        if action == 'SELL':
            swing_high = recent_data['high'].max()
            structure_stop = swing_high + buffer
            distance_pct = (structure_stop - entry_price) / entry_price * 100
            print(f"📊 Structure Stop (SHORT): ${structure_stop:.2f} ({distance_pct:.1f}% above entry)")
            return structure_stop
        else:
            swing_low = recent_data['low'].min()
            structure_stop = swing_low - buffer
            distance_pct = (entry_price - structure_stop) / entry_price * 100
            print(f"📊 Structure Stop (LONG): ${structure_stop:.2f} ({distance_pct:.1f}% below entry)")
            return structure_stop
    
    def detect_structure_break(self, df, current_position_side, entry_price, current_price):
        """Detect structure breaks during active positions"""
        lookback = 20
        
        if len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        buffer = entry_price * 0.002
        
        if current_position_side == 'sell':
            resistance_level = recent_data['high'].max()
            break_level = resistance_level + buffer
            
            if current_price > break_level:
                # Record this break for retest monitoring
                break_record = {
                    'type': 'resistance_break',
                    'level': resistance_level,
                    'break_price': current_price,
                    'break_time': df.index[-1] if len(df) > 0 else datetime.now(),
                    'direction': 'bullish'
                }
                self._record_structure_break(break_record)
                
                return {
                    'break_type': 'resistance_break',
                    'break_level': break_level,
                    'current_price': current_price,
                    'suggested_action': 'CLOSE_SHORT',
                    'flip_signal': 'BUY'
                }
                
        elif current_position_side == 'buy':
            support_level = recent_data['low'].min()
            break_level = support_level - buffer
            
            if current_price < break_level:
                # Record this break for retest monitoring
                break_record = {
                    'type': 'support_break',
                    'level': support_level,
                    'break_price': current_price,
                    'break_time': df.index[-1] if len(df) > 0 else datetime.now(),
                    'direction': 'bearish'
                }
                self._record_structure_break(break_record)
                
                return {
                    'break_type': 'support_break',
                    'break_level': break_level,
                    'current_price': current_price,
                    'suggested_action': 'CLOSE_LONG',
                    'flip_signal': 'SELL'
                }
        
        return None
    
    def _record_structure_break(self, break_record):
        """Record structure break for retest monitoring"""
        self.recent_breaks.append(break_record)
        
        # Keep only recent breaks
        if len(self.recent_breaks) > self.max_break_history:
            self.recent_breaks.pop(0)
        
        # Start monitoring for retest
        self.retest_monitoring = {
            'break_record': break_record,
            'monitoring_since': break_record['break_time'],
            'max_monitoring_bars': 15,  # Monitor for 15 bars (75 minutes on 5m chart)
            'retest_found': False
        }
        
        print(f"🔍 Break & Retest Monitor | {break_record['type']} @ ${break_record['level']:.2f}")
    
    def detect_break_and_retest(self, df, current_price):
        """
        NEW: Detect Break & Retest patterns after structure breaks
        Returns high-probability retest signals
        """
        if not self.retest_monitoring or len(df) < 10:
            return None
        
        break_record = self.retest_monitoring['break_record']
        break_level = break_record['level']
        break_direction = break_record['direction']
        break_time = break_record['break_time']
        
        # Check if monitoring period expired
        current_time = df.index[-1] if len(df) > 0 else datetime.now()
        bars_since_break = len(df) - len(df[df.index <= break_time]) if len(df) > 0 else 0
        
        if bars_since_break > self.retest_monitoring['max_monitoring_bars']:
            print(f"⏰ Retest Monitor Expired | No retest found within {bars_since_break} bars")
            self.retest_monitoring = None
            return None
        
        # Define retest zone (2% around break level)
        retest_buffer = break_level * 0.02
        retest_zone_high = break_level + retest_buffer
        retest_zone_low = break_level - retest_buffer
        
        # Check for pullback to retest zone
        in_retest_zone = retest_zone_low <= current_price <= retest_zone_high
        
        if not in_retest_zone:
            return None
        
        # Analyze recent price action for retest confirmation
        recent_bars = min(5, len(df))
        recent_data = df.tail(recent_bars)
        
        if break_direction == 'bullish':  # Resistance became support
            # Look for price holding above broken resistance
            lowest_recent = recent_data['low'].min()
            
            # Successful retest: price stays above break level
            if lowest_recent >= break_level * 0.998:  # 0.2% tolerance
                retest_strength = self._calculate_retest_strength(df, break_level, 'bullish')
                
                if retest_strength >= 0.6:  # 60% confidence threshold
                    self.retest_monitoring['retest_found'] = True
                    
                    # Calculate retest-specific stop (tighter)
                    retest_stop = break_level * 0.996  # Just below retested level
                    
                    print(f"✅ BULLISH RETEST | Level: ${break_level:.2f} | Strength: {retest_strength:.1%} | Stop: ${retest_stop:.2f}")
                    
                    return {
                        'action': 'BUY',
                        'price': current_price,
                        'rsi': 50.0,  # Neutral - pattern-based entry
                        'mfi': 50.0,
                        'timestamp': current_time,
                        'structure_stop': retest_stop,
                        'signal_type': 'BREAK_RETEST',
                        'retest_strength': retest_strength,
                        'break_level': break_level,
                        'pattern_type': 'bullish_retest'
                    }
        
        elif break_direction == 'bearish':  # Support became resistance
            # Look for price holding below broken support
            highest_recent = recent_data['high'].max()
            
            # Successful retest: price stays below break level
            if highest_recent <= break_level * 1.002:  # 0.2% tolerance
                retest_strength = self._calculate_retest_strength(df, break_level, 'bearish')
                
                if retest_strength >= 0.6:  # 60% confidence threshold
                    self.retest_monitoring['retest_found'] = True
                    
                    # Calculate retest-specific stop (tighter)
                    retest_stop = break_level * 1.004  # Just above retested level
                    
                    print(f"✅ BEARISH RETEST | Level: ${break_level:.2f} | Strength: {retest_strength:.1%} | Stop: ${retest_stop:.2f}")
                    
                    return {
                        'action': 'SELL',
                        'price': current_price,
                        'rsi': 50.0,  # Neutral - pattern-based entry
                        'mfi': 50.0,
                        'timestamp': current_time,
                        'structure_stop': retest_stop,
                        'signal_type': 'BREAK_RETEST',
                        'retest_strength': retest_strength,
                        'break_level': break_level,
                        'pattern_type': 'bearish_retest'
                    }
        
        return None
    
    def _calculate_retest_strength(self, df, break_level, direction):
        """Calculate retest pattern strength (0.0 to 1.0)"""
        if len(df) < 10:
            return 0.5
        
        recent_data = df.tail(10)
        strength_factors = []
        
        # Factor 1: Volume analysis (lower volume on pullback = good)
        if len(recent_data) >= 5:
            pullback_volume = recent_data.tail(3)['volume'].mean()
            breakout_volume = recent_data.head(3)['volume'].mean()
            volume_ratio = pullback_volume / breakout_volume if breakout_volume > 0 else 1.0
            
            # Lower volume on pullback is better (0.5-0.8 ratio ideal)
            if 0.5 <= volume_ratio <= 0.8:
                strength_factors.append(0.9)
            elif 0.3 <= volume_ratio <= 1.0:
                strength_factors.append(0.7)
            else:
                strength_factors.append(0.4)
        
        # Factor 2: Pullback depth (50-80% is ideal)
        if direction == 'bullish':
            highest_since_break = recent_data['high'].max()
            lowest_pullback = recent_data['low'].min()
            pullback_depth = (highest_since_break - lowest_pullback) / (highest_since_break - break_level)
        else:
            lowest_since_break = recent_data['low'].min()
            highest_pullback = recent_data['high'].max()
            pullback_depth = (highest_pullback - lowest_since_break) / (break_level - lowest_since_break)
        
        if 0.5 <= pullback_depth <= 0.8:
            strength_factors.append(0.9)
        elif 0.3 <= pullback_depth <= 0.9:
            strength_factors.append(0.7)
        else:
            strength_factors.append(0.5)
        
        # Factor 3: Time factor (retests within 5-10 bars are stronger)
        bars_in_retest = len(recent_data)
        if 5 <= bars_in_retest <= 10:
            strength_factors.append(0.8)
        elif 3 <= bars_in_retest <= 12:
            strength_factors.append(0.7)
        else:
            strength_factors.append(0.6)
        
        # Average all factors
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.6
    
    def calculate_indicators(self, df):
        df = df.copy()
        if len(df) < 2:
            return df
        
        df['rsi'] = self.calculate_rsi(df['close'])
        df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
        
        return df
    
    def generate_signal(self, df):
        """Enhanced signal generation with Break & Retest priority"""
        min_bars = max(self.params['rsi_length'], self.params['mfi_length']) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        
        # PRIORITY 1: Check for Break & Retest patterns (highest probability)
        retest_signal = self.detect_break_and_retest(df, current_price)
        if retest_signal:
            return retest_signal
        
        # PRIORITY 2: Original RSI/MFI signals (ranging markets)
        current_rsi = df['rsi'].iloc[-1]
        current_mfi = df['mfi'].iloc[-1]
        
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None
        
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        
        # BUY signal - Both RSI and MFI oversold
        if (current_rsi < oversold and 
            current_mfi < oversold and 
            self.last_signal != 'BUY'):
            
            self.last_signal = 'BUY'
            structure_stop = self.get_structure_stop(df, 'BUY', current_price)
            
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1],
                'structure_stop': structure_stop,
                'signal_type': 'RSI_MFI'
            }
        
        # SELL signal - Both RSI and MFI overbought
        elif (current_rsi > overbought and 
              current_mfi > overbought and 
              self.last_signal != 'SELL'):
            
            self.last_signal = 'SELL'
            structure_stop = self.get_structure_stop(df, 'SELL', current_price)
            
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1],
                'structure_stop': structure_stop,
                'signal_type': 'RSI_MFI'
            }
        
        return None