import json
import os
import pandas as pd
import numpy as np

class RSIMFICloudStrategy:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # ATR Risk Management (optimized for ZORA volatility)
        self.atr_period = 14
        self.atr_multiplier = 1.5          # Reduced from 2.0 for tighter stops
        
        # Signal Management
        self.signal_cooldown_period = 5    # Bars to wait between signals
        
        # Load strategy parameters
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
        with open(params_file, 'r') as f:
            self.params = json.load(f)
        
        # Runtime variables
        self.last_signal = None
        self.signal_cooldown = 0
        self.trailing_stop = None
        self.position_type = None
        self.entry_price = None
        
    def calculate_rsi(self, prices, period=7):
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return pd.Series([50] * len(prices), index=prices.index)
                
            prices = pd.Series(prices).astype(float)
            deltas = prices.diff()
            
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            avg_gain = gains.ewm(span=period, adjust=False).mean()
            avg_loss = losses.ewm(span=period, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, 0.0001)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).replace([np.inf, -np.inf], 50).clip(0, 100)
            
        except Exception as e:
            print(f"RSI error: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_mfi(self, high, low, close, period=7):
        """Calculate MFI without volume"""
        try:
            if len(close) < period + 1:
                return pd.Series([50] * len(close), index=close.index)
                
            high = pd.Series(high).astype(float)
            low = pd.Series(low).astype(float)
            close = pd.Series(close).astype(float)
            
            typical_price = (high + low + close) / 3
            price_range = high - low
            money_flow = typical_price * price_range
            
            mf_sign = typical_price.diff()
            positive_mf = money_flow.where(mf_sign > 0, 0)
            negative_mf = money_flow.where(mf_sign <= 0, 0)
            
            positive_mf_ema = positive_mf.ewm(span=period, adjust=False).mean()
            negative_mf_ema = negative_mf.ewm(span=period, adjust=False).mean()
            
            mf_ratio = positive_mf_ema / negative_mf_ema.replace(0, 0.0001)
            mfi = 100 - (100 / (1 + mf_ratio))
            
            return mfi.fillna(50).replace([np.inf, -np.inf], 50).clip(0, 100)
            
        except Exception as e:
            print(f"MFI error: {e}")
            return pd.Series([50] * len(close), index=close.index)
    
    def calculate_trend(self, close):
        """Calculate trend efficiently using vectorized operations"""
        try:
            ema_fast = close.ewm(span=12).mean()
            ema_slow = close.ewm(span=26).mean()
            
            # Vectorized calculations
            ema_distance_pct = abs(ema_fast - ema_slow) / close * 100
            min_separation = 0.15
            
            # Create trend series
            trend = pd.Series(['SIDEWAYS'] * len(close), index=close.index)
            
            uptrend_mask = (ema_fast > ema_slow) & (ema_distance_pct >= min_separation)
            downtrend_mask = (ema_fast < ema_slow) & (ema_distance_pct >= min_separation)
            
            trend[uptrend_mask] = 'UP'
            trend[downtrend_mask] = 'DOWN'
            
            return trend
                
        except Exception as e:
            print(f"Trend error: {e}")
            return pd.Series(['SIDEWAYS'] * len(close), index=close.index)
    
    def calculate_atr(self, df):
        """Calculate ATR manually"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # Calculate True Range
            high_low = high - low
            high_close = np.abs(high - close.shift(1))
            low_close = np.abs(low - close.shift(1))
            
            true_range = pd.DataFrame({
                'hl': high_low,
                'hc': high_close,
                'lc': low_close
            }).max(axis=1)
            
            # Calculate ATR using EMA
            atr = true_range.ewm(span=self.atr_period, adjust=False).mean()
            
            return atr.fillna(0)
            
        except Exception as e:
            print(f"ATR error: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def set_position(self, position_type, entry_price, current_atr):
        """Initialize position with ATR-based stop"""
        self.position_type = position_type
        self.entry_price = entry_price
        
        if position_type == 'LONG':
            self.trailing_stop = entry_price - (current_atr * self.atr_multiplier)
        elif position_type == 'SHORT':
            self.trailing_stop = entry_price + (current_atr * self.atr_multiplier)
    
    def update_trailing_stop(self, current_price, current_atr):
        """Update trailing stop based on current price and ATR"""
        if self.position_type is None or self.trailing_stop is None:
            return self.trailing_stop
            
        if self.position_type == 'LONG':
            new_stop = current_price - (current_atr * self.atr_multiplier)
            self.trailing_stop = max(self.trailing_stop, new_stop)
        elif self.position_type == 'SHORT':
            new_stop = current_price + (current_atr * self.atr_multiplier)
            self.trailing_stop = min(self.trailing_stop, new_stop)
            
        return self.trailing_stop
    
    def check_stop_hit(self, current_price):
        """Check if trailing stop is hit"""
        if self.trailing_stop is None or self.position_type is None:
            return False
            
        if self.position_type == 'LONG':
            return current_price <= self.trailing_stop
        elif self.position_type == 'SHORT':
            return current_price >= self.trailing_stop
            
        return False
    
    def reset_position(self):
        """Clear position data"""
        self.trailing_stop = None
        self.position_type = None
        self.entry_price = None
        
    def calculate_indicators(self, df):
        """Calculate all indicators"""
        df = df.copy()
        
        if len(df) < 2:
            return df
            
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_length'])
        df['mfi'] = self.calculate_mfi(
            df['high'], df['low'], df['close'],
            self.params['mfi_length']
        )
        df['trend'] = self.calculate_trend(df['close'])
        df['price_change'] = df['close'].pct_change(fill_method=None)
        df['atr'] = self.calculate_atr(df)
        
        return df
    
    def generate_signal(self, df):
        """Generate trading signals with ATR-based risk management"""
        min_bars = max(self.params['rsi_length'], self.params['mfi_length'], 26, self.atr_period) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current values
        current_rsi = df['rsi'].iloc[-1]
        current_mfi = df['mfi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_trend = df['trend'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        # Validate
        if pd.isna(current_rsi) or pd.isna(current_mfi) or pd.isna(current_price) or pd.isna(current_atr):
            return None
        
        # Check if current position should be closed (trailing stop)
        if self.position_type:
            self.update_trailing_stop(current_price, current_atr)
            
            if self.check_stop_hit(current_price):
                self.reset_position()
                return {
                    'action': 'CLOSE',
                    'reason': 'TRAILING_STOP_HIT',
                    'price': current_price,
                    'stop_price': self.trailing_stop,
                    'timestamp': df.index[-1]
                }
        
        # Signal cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None
        
        # Signal conditions
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        
        buy_conditions = [
            current_rsi < oversold,
            current_mfi < oversold,
            current_trend == "UP" if self.params['require_trend'] else True,
            self.last_signal != 'BUY'
        ]
        
        sell_conditions = [
            current_rsi > overbought,
            current_mfi > overbought,
            current_trend == "DOWN" if self.params['require_trend'] else True,
            self.last_signal != 'SELL'
        ]
        
        # Generate signals
        if all(buy_conditions):
            self.last_signal = 'BUY'
            self.signal_cooldown = self.signal_cooldown_period
            self.set_position('LONG', current_price, current_atr)
            
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1],
                'confidence': 'TREND_FILTERED',
                'trailing_stop': round(self.trailing_stop, 4),
                'atr': round(current_atr, 4)
            }
            
        elif all(sell_conditions):
            self.last_signal = 'SELL'
            self.signal_cooldown = self.signal_cooldown_period
            self.set_position('SHORT', current_price, current_atr)
            
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1],
                'confidence': 'TREND_FILTERED',
                'trailing_stop': round(self.trailing_stop, 4),
                'atr': round(current_atr, 4)
            }
        
        return None