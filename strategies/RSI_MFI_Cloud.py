import json
import os
import pandas as pd
import numpy as np

class RSIMFICloudStrategy:
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager  # Get symbol from risk manager
        self._load_config()
        self.last_signal = None
    
    def _load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
        with open(params_file, 'r') as f:
            self.params = json.load(f)
    
    @property
    def symbol(self):
        # Get symbol from risk manager, not from params
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
        money_flow = typical_price * volume  # Use volume, not price range
        
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
        lookback = 20  # Look back 20 bars (100 minutes on 5m chart)
        
        if len(df) < lookback:
            # Fallback to fixed % if not enough data
            buffer = entry_price * 0.002  # 0.2% buffer
            if action == 'SELL':
                fixed_stop = entry_price * 1.015 + buffer  # 1.5% above entry + buffer
                print(f"📊 Using Fixed Stop (insufficient data): ${fixed_stop:.2f}")
                return fixed_stop
            else:
                fixed_stop = entry_price * 0.985 - buffer  # 1.5% below entry - buffer
                print(f"📊 Using Fixed Stop (insufficient data): ${fixed_stop:.2f}")
                return fixed_stop
        
        recent_data = df.tail(lookback)
        buffer = entry_price * 0.002  # 0.2% buffer to avoid wicks
        
        if action == 'SELL':  # Short position
            # Stop above recent swing high
            swing_high = recent_data['high'].max()
            structure_stop = swing_high + buffer
            distance_pct = (structure_stop - entry_price) / entry_price * 100
            print(f"📊 Structure Stop (SHORT): ${structure_stop:.2f} ({distance_pct:.1f}% above entry)")
            return structure_stop
        else:  # Long position  
            # Stop below recent swing low
            swing_low = recent_data['low'].min()
            structure_stop = swing_low - buffer
            distance_pct = (entry_price - structure_stop) / entry_price * 100
            print(f"📊 Structure Stop (LONG): ${structure_stop:.2f} ({distance_pct:.1f}% below entry)")
            return structure_stop
    
    def detect_structure_break(self, df, current_position_side, entry_price, current_price):
        """
        NEW: Detect structure breaks during active positions
        - For SHORT positions: Watch for breaks ABOVE recent resistance
        - For LONG positions: Watch for breaks BELOW recent support
        """
        lookback = 20  # Same as structure stop calculation
        
        if len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        buffer = entry_price * 0.002  # 0.2% buffer to avoid false signals
        
        if current_position_side == 'sell':  # SHORT position
            # Check for break above resistance (swing high)
            resistance_level = recent_data['high'].max()
            break_level = resistance_level + buffer
            
            if current_price > break_level:
                return {
                    'break_type': 'resistance_break',
                    'break_level': break_level,
                    'current_price': current_price,
                    'suggested_action': 'CLOSE_SHORT',
                    'flip_signal': 'BUY'  # Optional: flip to long
                }
                
        elif current_position_side == 'buy':  # LONG position
            # Check for break below support (swing low)
            support_level = recent_data['low'].min()
            break_level = support_level - buffer
            
            if current_price < break_level:
                return {
                    'break_type': 'support_break',
                    'break_level': break_level,
                    'current_price': current_price,
                    'suggested_action': 'CLOSE_LONG',
                    'flip_signal': 'SELL'  # Optional: flip to short
                }
        
        return None
    
    def calculate_indicators(self, df):
        df = df.copy()
        if len(df) < 2:
            return df
        
        df['rsi'] = self.calculate_rsi(df['close'])
        df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])  # Fixed: pass volume
        
        return df
    
    def generate_signal(self, df):
        min_bars = max(self.params['rsi_length'], self.params['mfi_length']) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current values
        current_rsi = df['rsi'].iloc[-1]
        current_mfi = df['mfi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Check for invalid values
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None
        
        # Signal conditions
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        
        # BUY signal - Both RSI and MFI oversold
        if (current_rsi < oversold and 
            current_mfi < oversold and 
            self.last_signal != 'BUY'):
            
            self.last_signal = 'BUY'
            
            # Calculate structure-based stop
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
            
            # Calculate structure-based stop
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