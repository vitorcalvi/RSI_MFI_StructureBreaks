import pandas as pd
import numpy as np
from datetime import datetime

class RSIMFIStrategy:
    def __init__(self):
        self.config = {
            "rsi_length": 2,                    # 3 → 2 (faster response)
            "mfi_length": 3,
            "uptrend_oversold": 48,             # Keep exactly as requested
            "downtrend_overbought": 52,         # Keep exactly as requested
            "neutral_oversold": 45,
            "neutral_overbought": 55,
            "cooldown_seconds": 0.5             # 1s → 0.5s (faster signals)
        }
        self.last_signal_time = None
    
    def calculate_rsi(self, prices):
        """Calculate RSI indicator - streamlined"""
        period = self.config['rsi_length']
        if len(prices) < period + 1:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return (100 - 100 / (1 + rs)).fillna(50.0).clip(0, 100)
    
    def calculate_mfi(self, high, low, close, volume):
        """Calculate Money Flow Index - streamlined"""
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
        """Detect market trend - simplified logic"""
        if len(data) < 50:
            return 'neutral'
        
        close = data['close']
        ema10, ema21, ema50 = close.ewm(span=10).mean().iloc[-1], close.ewm(span=21).mean().iloc[-1], close.ewm(span=50).mean().iloc[-1]
        current_price, price_5_ago = close.iloc[-1], close.iloc[-5]
        
        # Streamlined trend detection
        if ema10 > ema21 > ema50 and current_price > ema10 and current_price > price_5_ago * 1.001:
            return 'strong_uptrend'
        if ema10 < ema21 < ema50 and current_price < ema10 and current_price < price_5_ago * 0.999:
            return 'strong_downtrend'
        
        return 'neutral'
    
    def generate_signal(self, data):
        """Generate trading signals - streamlined with aggressive MFI thresholds"""
        if len(data) < 50 or self._is_cooldown_active():
            return None
        
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        trend = self.detect_trend(data)
        price = data['close'].iloc[-1]
        
        # Streamlined signal generation with aggressive MFI (30/70 thresholds)
        signal = None
        if trend == 'strong_uptrend' and rsi <= self.config['uptrend_oversold'] and mfi <= 70:  # 50 → 70
            signal = self._create_signal('BUY', trend, rsi, mfi, price, data)
        elif trend == 'strong_downtrend' and rsi >= self.config['downtrend_overbought'] and mfi >= 30:  # 50 → 30
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
        """Create signal with tighter structure stops for scalping"""
        window = data.tail(50)
        
        # Tighter stops: 0.998/1.002 → 0.9985/1.0015
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.9985
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.0015
            level = window['high'].max()
        
        return {
            'action': action, 'trend': trend, 'rsi': rsi, 'mfi': mfi,
            'price': price, 'structure_stop': structure_stop, 'level': level,
            'signal_type': f"{trend}_{action.lower()}",
            'confidence': min(95, max(70, abs(50 - rsi) + abs(50 - mfi)))
        }
    
    def _is_cooldown_active(self):
        """Check cooldown - streamlined"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def calculate_indicators(self, data):
        """Calculate indicators - streamlined"""
        if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 1:
            return {}
        
        try:
            return {
                'rsi': self.calculate_rsi(data['close']),
                'mfi': self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            }
        except:
            return {}
    
    def get_strategy_info(self):
        """Get strategy info - streamlined"""
        return {'name': 'RSI/MFI Trend Following', 'config': self.config}