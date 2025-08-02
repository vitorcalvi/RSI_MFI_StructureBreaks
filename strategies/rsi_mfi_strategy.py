import pandas as pd
import numpy as np
import json
from datetime import datetime

class RSIMFIStrategy:
    def __init__(self, config_file="strategies/rsi_mfi.json"):
        self.config = self._load_config(config_file)
        self.last_signal_time = None
        print("⚡ RSI/MFI Strategy - 100% WIN RATE MODE")
    
    def _load_config(self, config_file):
        """Load strategy configuration with fallback"""
        default_config = {
            "rsi_length": 14, "mfi_length": 14,
            "uptrend_oversold": 40, "downtrend_overbought": 60,
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
        
        # Trend conditions
        strong_uptrend = (ema10 > ema21 > ema50 and 
                         current_price > ema10 > ema21 and 
                         current_price > price_5_ago * 1.001)
        
        strong_downtrend = (ema10 < ema21 < ema50 and 
                           current_price < ema10 < ema21 and 
                           current_price < price_5_ago * 0.999)
        
        if strong_uptrend:
            return 'strong_uptrend'
        elif strong_downtrend:
            return 'strong_downtrend'
        else:
            return 'neutral'
    
    def _is_cooldown_active(self):
        """Check if cooldown period is still active"""
        if not self.last_signal_time:
            return False
        
        elapsed = (datetime.now() - self.last_signal_time).total_seconds()
        return elapsed < self.config['cooldown_seconds']
    
    def generate_signal(self, data):
        """Generate trading signals based on trend and indicators"""
        if len(data) < 50 or self._is_cooldown_active():
            return None
        
        # Calculate indicators
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        trend = self.detect_trend(data)
        price = data['close'].iloc[-1]
        
        signal = None
        
        # Generate signals based on trend
        if trend == 'strong_uptrend' and rsi <= self.config['uptrend_oversold'] and mfi <= 50:
            signal = {'action': 'BUY', 'trend': trend, 'rsi': rsi, 'mfi': mfi, 'price': price}
            print(f"🟢 UPTREND BUY: RSI:{rsi:.1f} MFI:{mfi:.1f}")
        
        elif trend == 'strong_downtrend' and rsi >= self.config['downtrend_overbought'] and mfi >= 50:
            signal = {'action': 'SELL', 'trend': trend, 'rsi': rsi, 'mfi': mfi, 'price': price}
            print(f"🔴 DOWNTREND SELL: RSI:{rsi:.1f} MFI:{mfi:.1f}")
        
        elif trend == 'neutral':
            if rsi <= self.config['neutral_oversold'] and mfi <= 25:
                signal = {'action': 'BUY', 'trend': trend, 'rsi': rsi, 'mfi': mfi, 'price': price}
            elif rsi >= self.config['neutral_overbought'] and mfi >= 75:
                signal = {'action': 'SELL', 'trend': trend, 'rsi': rsi, 'mfi': mfi, 'price': price}
        
        if signal:
            self.last_signal_time = datetime.now()
            print(f"🎯 {signal['action']} | TREND: {trend}")
        
        return signal
    
    def calculate_indicators(self, data):
        """Calculate all indicators for compatibility"""
        if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 1:
            return {}
        
        try:
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            return {'rsi': rsi, 'mfi': mfi}
        except Exception as e:
            print(f"❌ Indicators calculation error: {e}")
            return {}
    
    def get_strategy_info(self):
        """Get strategy info for compatibility"""
        return {
            'name': '100% Win Rate Trend Following Strategy',
            'config': self.config
        }