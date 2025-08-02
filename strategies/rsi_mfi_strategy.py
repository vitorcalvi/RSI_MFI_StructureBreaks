import pandas as pd
import numpy as np
from datetime import datetime

class RSIMFIStrategy:
    def __init__(self):
        self.config = {
            "rsi_length": 5,                    # 2 → 5 (fast but stable)
            "mfi_length": 5,                    # 3 → 5 (fast but stable) 
            "uptrend_oversold": 45,             # Scalping levels
            "downtrend_overbought": 55,         # Scalping levels
            "neutral_oversold": 40,             # Aggressive for HF
            "neutral_overbought": 60,           # Aggressive for HF
            "cooldown_seconds": 0.5             # Fast signals
        }
        self.last_signal_time = None
    
    def calculate_rsi(self, prices):
        """HF Scalping RSI - fast but stable"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        # Use EWM but with alpha smoothing to prevent extremes
        alpha = 2.0 / (period + 1)
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
        
        # Prevent extreme values with better division handling
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Clip to reasonable scalping range (not 0-100 extremes)
        return rsi.fillna(50.0).clip(5, 95)
    
    def calculate_mfi(self, high, low, close, volume):
        """HF Scalping MFI - fast response with bounds"""
        period = self.config['mfi_length']
        if len(close) < period + 5:
            return pd.Series(50.0, index=close.index)
        
        # Handle zero volume (common in scalping timeframes)
        if volume.sum() == 0:
            return pd.Series(50.0, index=close.index)
        
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        # Price direction with smoothing
        mf_change = tp.diff()
        pos_mf = money_flow.where(mf_change > 0, 0)
        neg_mf = money_flow.where(mf_change <= 0, 0)
        
        # Fast EWM for scalping responsiveness
        alpha = 2.0 / (period + 1)
        pos_mf_avg = pos_mf.ewm(alpha=alpha, min_periods=period).mean()
        neg_mf_avg = neg_mf.ewm(alpha=alpha, min_periods=period).mean()
        
        # Prevent 0/100 extremes while keeping sensitivity
        mfi_ratio = pos_mf_avg / (neg_mf_avg + 1e-8)
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        # Scalping bounds - not too restrictive
        return mfi.fillna(50.0).clip(10, 90)
    
    def detect_trend(self, data):
        """HF Scalping trend - fast detection"""
        if len(data) < 20:  # Reduced from 50 for HF
            return 'neutral'
        
        close = data['close']
        
        # Fast EMAs for scalping
        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        
        current_price = close.iloc[-1]
        
        # Quick momentum check (3 periods for HF)
        momentum = (current_price - close.iloc[-3]) / close.iloc[-3]
        
        # Fast trend detection for scalping
        if ema5 > ema10 > ema20 and current_price > ema5 and momentum > 0.0005:
            return 'strong_uptrend'
        if ema5 < ema10 < ema20 and current_price < ema5 and momentum < -0.0005:
            return 'strong_downtrend'
        
        return 'neutral'
    
    def generate_signal(self, data):
        """HF Scalping signals - aggressive but filtered"""
        if len(data) < 20 or self._is_cooldown_active():  # Reduced data requirement
            return None
        
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        trend = self.detect_trend(data)
        price = data['close'].iloc[-1]
        
        # Quick validation for HF scalping
        if pd.isna(rsi) or pd.isna(mfi):
            return None
        
        # HF Scalping signal logic - more aggressive
        signal = None
        
        if trend == 'strong_uptrend':
            # Buy dips in uptrend
            if rsi <= self.config['uptrend_oversold'] and mfi <= 60:
                signal = self._create_signal('BUY', trend, rsi, mfi, price, data)
                
        elif trend == 'strong_downtrend':
            # Sell rallies in downtrend  
            if rsi >= self.config['downtrend_overbought'] and mfi >= 40:
                signal = self._create_signal('SELL', trend, rsi, mfi, price, data)
                
        elif trend == 'neutral':
            # Scalp reversals in neutral
            if rsi <= self.config['neutral_oversold'] and mfi <= 35:
                signal = self._create_signal('BUY', trend, rsi, mfi, price, data)
            elif rsi >= self.config['neutral_overbought'] and mfi >= 65:
                signal = self._create_signal('SELL', trend, rsi, mfi, price, data)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action, trend, rsi, mfi, price, data):
        """HF Scalping signal creation - tight stops"""
        window = data.tail(20)  # Shorter window for HF
        
        # Tight stops for scalping (0.15% typical)
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.9985
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.0015
            level = window['high'].max()
        
        # Validate stop for scalping (0.05% to 0.5%)
        stop_distance = abs(price - structure_stop) / price
        if stop_distance < 0.0005 or stop_distance > 0.005:
            return None
        
        # HF confidence based on indicator divergence from 50
        rsi_strength = abs(50 - rsi)
        mfi_strength = abs(50 - mfi)
        confidence = min(95, max(70, (rsi_strength + mfi_strength) * 2))
        
        return {
            'action': action,
            'trend': trend,
            'rsi': round(rsi, 1),
            'mfi': round(mfi, 1),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"{trend}_{action.lower()}",
            'confidence': round(confidence, 1)
        }
    
    def _is_cooldown_active(self):
        """HF cooldown check"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def calculate_indicators(self, data):
        """HF indicator calculation"""
        if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 5:
            return {}
        
        try:
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            
            # Quick validation
            if rsi.isna().all() or mfi.isna().all():
                return {}
                
            return {'rsi': rsi, 'mfi': mfi}
        except:
            return {}
    
    def get_strategy_info(self):
        """Strategy info"""
        return {'name': 'RSI/MFI HF Scalping', 'config': self.config}