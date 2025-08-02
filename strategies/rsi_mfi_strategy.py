import pandas as pd
import numpy as np
import json
from datetime import datetime

class RSIMFIStrategy:
    def __init__(self, config_file="strategies/rsi_mfi.json"):
        self.config = self._load_config(config_file)
        self.last_signal_time = None
    
    def _load_config(self, config_file):
        """Load strategy configuration"""
        default_config = {
            "rsi_length": 14, "mfi_length": 14,
            "uptrend_oversold": 40, "downtrend_overbought": 50,
            "neutral_oversold": 25, "neutral_overbought": 75,
            "cooldown_seconds": 5
        }
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            return default_config
    
    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        period = self.config['rsi_length']
        if len(prices) < period + 1:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return (100 - 100 / (1 + rs)).fillna(50.0).clip(0, 100)
    
    def calculate_mfi(self, high, low, close, volume):
        """Calculate Money Flow Index"""
        period = self.config['mfi_length']
        if len(close) < period + 1:
            return pd.Series(50.0, index=close.index)
        
        tp = (high + low + close) / 3
        money_flow = tp * volume
        mf_sign = tp.diff()
        
        positive_mf = money_flow.where(mf_sign > 0, 0).rolling(period).sum()
        negative_mf = money_flow.where(mf_sign <= 0, 0).rolling(period).sum()
        mfi_ratio = positive_mf / negative_mf.replace(0, 1e-10)
        return (100 - 100 / (1 + mfi_ratio)).fillna(50.0).clip(0, 100)
    
    def detect_trend(self, data):
        """Detect market trend using EMAs"""
        if len(data) < 50:
            return 'neutral'
        
        close = data['close']
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        current_price = close.iloc[-1]
        price_5_ago = close.iloc[-5]
        
        # Strong uptrend
        if (ema10 > ema21 > ema50 and 
            current_price > ema10 > ema21 and 
            current_price > price_5_ago * 1.001):
            return 'strong_uptrend'
        
        # Strong downtrend
        if (ema10 < ema21 < ema50 and 
            current_price < ema10 < ema21 and 
            current_price < price_5_ago * 0.999):
            return 'strong_downtrend'
        
        return 'neutral'
    
    def _is_cooldown_active(self):
        """Check if cooldown period is active"""
        if not self.last_signal_time:
            return False
        elapsed = (datetime.now() - self.last_signal_time).total_seconds()
        return elapsed < self.config['cooldown_seconds']
    
    def generate_signal(self, data):
        """Generate trading signals"""
        if len(data) < 50 or self._is_cooldown_active():
            return None
        
        # Calculate indicators
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        trend = self.detect_trend(data)
        price = data['close'].iloc[-1]
        
        # Generate signals based on trend
        signal = None
        if trend == 'strong_uptrend' and rsi <= self.config['uptrend_oversold'] and mfi <= 50:
            signal = self._create_signal('BUY', trend, rsi, mfi, price, data)
        elif trend == 'strong_downtrend' and rsi >= self.config['downtrend_overbought'] and mfi >= 50:
            signal = self._create_signal('SELL', trend, rsi, mfi, price, data)
        elif trend == 'neutral':
            if rsi <= self.config['neutral_oversold'] and mfi <= 25:
                signal = self._create_signal('BUY', trend, rsi, mfi, price, data)
            elif rsi >= self.config['neutral_overbought'] and mfi >= 75:
                signal = self._create_signal('SELL', trend, rsi, mfi, price, data)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action, trend, rsi, mfi, price, data):
        """Create signal with structure stop"""
        window = data.tail(50)
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.998
            level = window['low'].min()
            signal_type = f"{trend}_buy"
        else:
            structure_stop = window['high'].max() * 1.002
            level = window['high'].max()
            signal_type = f"{trend}_sell"
        
        return {
            'action': action,
            'trend': trend,
            'rsi': rsi,
            'mfi': mfi,
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': signal_type,
            'confidence': min(95, max(70, abs(50 - rsi) + abs(50 - mfi)))
        }
    
    def calculate_indicators(self, data):
        """Calculate all indicators"""
        if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 1:
            return {}
        
        try:
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            return {'rsi': rsi, 'mfi': mfi}
        except:
            return {}
    
    def get_strategy_info(self):
        """Get strategy info"""
        return {
            'name': 'RSI/MFI Trend Following',
            'config': self.config
        }