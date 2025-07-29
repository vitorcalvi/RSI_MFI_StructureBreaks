import json
import os
import pandas as pd
import numpy as np

class RSIMFICloudStrategy:
    def __init__(self):
        self._load_config()
        self.last_signal = None
        self.signal_cooldown = 0
    
    def _load_config(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
            with open(params_file, 'r') as f:
                self.params = json.load(f)
        except:
            self.params = {
                'symbol': 'ZORA/USDT',
                'rsi_length': 5,
                'mfi_length': 5,
                'oversold_level': 40,
                'overbought_level': 50,
                'signal_cooldown': 1,
                'require_trend': False
            }
    
    @property
    def symbol(self):
        return self.params.get('symbol', 'ZORA/USDT')
    
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
    
    def calculate_mfi(self, high, low, close):
        period = self.params['mfi_length']
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
        
        positive_mf_ema = positive_mf.ewm(span=period).mean()
        negative_mf_ema = negative_mf.ewm(span=period).mean()
        
        mf_ratio = positive_mf_ema / negative_mf_ema.replace(0, 0.0001)
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi.fillna(50).clip(0, 100)
    
    def calculate_trend(self, close):
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        
        trend = pd.Series(['SIDEWAYS'] * len(close), index=close.index)
        trend[ema_fast > ema_slow] = 'UP'
        trend[ema_fast < ema_slow] = 'DOWN'
        
        return trend
    
    def calculate_indicators(self, df):
        df = df.copy()
        if len(df) < 2:
            return df
        
        df['rsi'] = self.calculate_rsi(df['close'])
        df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'])
        df['trend'] = self.calculate_trend(df['close'])
        
        return df
    
    def generate_signal(self, df):
        min_bars = max(self.params['rsi_length'], self.params['mfi_length'], 26) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current values
        current_rsi = df['rsi'].iloc[-1]
        current_mfi = df['mfi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_trend = df['trend'].iloc[-1]
        
        # Check for invalid values
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None
        
        # Check cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None
        
        # Signal conditions
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        require_trend = self.params.get('require_trend', False)
        
        # BUY signal
        if (current_rsi < oversold and current_mfi < oversold and 
            (not require_trend or current_trend == "UP") and 
            self.last_signal != 'BUY'):
            
            self.last_signal = 'BUY'
            self.signal_cooldown = self.params['signal_cooldown']
            
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1]
            }
        
        # SELL signal
        elif (current_rsi > overbought and current_mfi > overbought and 
              (not require_trend or current_trend == "DOWN") and 
              self.last_signal != 'SELL'):
            
            self.last_signal = 'SELL'
            self.signal_cooldown = self.params['signal_cooldown']
            
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1]
            }
        
        return None