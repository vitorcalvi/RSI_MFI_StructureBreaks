import json
import os
import pandas as pd
import numpy as np

class RSIMFICloudStrategy:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # ATR Configuration
        self.atr_period = 14
        
        # Signal Management
        self.signal_cooldown_period = 5
        
        # Load strategy parameters
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
        with open(params_file, 'r') as f:
            self.params = json.load(f)
        
        # Runtime variables (ONLY for strategy logic)
        self.last_signal = None
        self.signal_cooldown = 0
        self.current_atr_pct = 0  # For risk management integration
        
    # ==========================================
    # INDICATOR CALCULATIONS (Single Responsibility)
    # ==========================================
    
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
    
    # ==========================================
    # INDICATOR AGGREGATION (Single Responsibility)
    # ==========================================
    
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
    
    # ==========================================
    # SIGNAL VALIDATION (Single Responsibility)
    # ==========================================
    
    def validate_signal_conditions(self, current_rsi, current_mfi, current_trend, oversold, overbought):
        """Validate signal conditions"""
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
        
        return buy_conditions, sell_conditions
    
    def check_signal_cooldown(self):
        """Check and update signal cooldown"""
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return False
        return True
    
    def update_signal_state(self, signal_type):
        """Update signal state after signal generation"""
        self.last_signal = signal_type
        self.signal_cooldown = self.signal_cooldown_period
    
    # ==========================================
    # SIGNAL GENERATION (Single Responsibility)
    # ==========================================
    
    def generate_signal(self, df):
        """Generate trading signals - PURE STRATEGY LOGIC ONLY"""
        min_bars = max(self.params['rsi_length'], self.params['mfi_length'], 26, self.atr_period) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Extract current values
        current_values = self.extract_current_values(df)
        if not current_values:
            return None
        
        current_rsi, current_mfi, current_price, current_trend, current_atr = current_values
        
        # Update ATR percentage for risk management integration
        self.update_atr_percentage(current_price, current_atr)
        
        # Check cooldown
        if not self.check_signal_cooldown():
            return None
        
        # Generate signals
        return self.evaluate_signal_conditions(current_rsi, current_mfi, current_price, current_trend, df.index[-1])
    
    def extract_current_values(self, df):
        """Extract current indicator values"""
        try:
            current_rsi = df['rsi'].iloc[-1]
            current_mfi = df['mfi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            current_trend = df['trend'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            # Validate
            if pd.isna(current_rsi) or pd.isna(current_mfi) or pd.isna(current_price) or pd.isna(current_atr):
                return None
            
            return current_rsi, current_mfi, current_price, current_trend, current_atr
            
        except (IndexError, KeyError):
            return None
    
    def update_atr_percentage(self, current_price, current_atr):
        """Update ATR percentage for risk management integration"""
        if current_price > 0 and current_atr > 0:
            self.current_atr_pct = (current_atr / current_price) * 100
        else:
            self.current_atr_pct = 0
    
    def evaluate_signal_conditions(self, current_rsi, current_mfi, current_price, current_trend, timestamp):
        """Evaluate signal conditions and generate signal"""
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        
        buy_conditions, sell_conditions = self.validate_signal_conditions(
            current_rsi, current_mfi, current_trend, oversold, overbought
        )
        
        # Generate BUY signal
        if all(buy_conditions):
            self.update_signal_state('BUY')
            return self.create_signal_data('BUY', current_price, current_rsi, current_mfi, current_trend, timestamp)
            
        # Generate SELL signal
        elif all(sell_conditions):
            self.update_signal_state('SELL')
            return self.create_signal_data('SELL', current_price, current_rsi, current_mfi, current_trend, timestamp)
        
        return None
    
    def create_signal_data(self, action, price, rsi, mfi, trend, timestamp):
        """Create signal data structure"""
        return {
            'action': action,
            'price': price,
            'rsi': round(rsi, 2),
            'mfi': round(mfi, 2),
            'trend': trend,
            'timestamp': timestamp,
            'confidence': 'TREND_FILTERED' if self.params['require_trend'] else 'STANDARD',
            'atr_pct': round(self.current_atr_pct, 2)
        }