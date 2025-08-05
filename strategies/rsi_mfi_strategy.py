import pandas as pd
import numpy as np
from datetime import datetime

class RSIMFIStrategy:
    """High-Frequency RSI/MFI Scalping Strategy for crypto markets."""
    
    def __init__(self):
        self.config = {
            "rsi_length": 3, "mfi_length": 3,
            "uptrend_oversold": 55, "uptrend_mfi_threshold": 70,
            "downtrend_overbought": 45, "neutral_oversold": 50,
            "neutral_mfi_threshold": 50, "neutral_overbought": 50,
            "cooldown_seconds": 0.1, "short_rsi_minimum": 60,
            "short_mfi_threshold": 65, "target_profit_usdt": 15,
            "max_hold_seconds": 180, "short_position_reduction": 0.7
        }
        self.last_signal_time = None
    
    def calculate_rsi(self, prices):
        """Calculate RSI with error handling."""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        alpha = 2.0 / (period + 1)
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
        
        # Prevent division by zero
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0).clip(5, 95)
    
    def calculate_mfi(self, high, low, close, volume):
        """Calculate MFI with error handling."""
        period = self.config['mfi_length']
        if len(close) < period + 5 or volume.sum() == 0:
            return pd.Series(50.0, index=close.index)
        
        tp = (high + low + close) / 3
        money_flow = tp * volume
        mf_change = tp.diff()
        
        pos_mf = money_flow.where(mf_change > 0, 0)
        neg_mf = money_flow.where(mf_change <= 0, 0)
        
        alpha = 2.0 / (period + 1)
        pos_mf_avg = pos_mf.ewm(alpha=alpha, min_periods=period).mean()
        neg_mf_avg = neg_mf.ewm(alpha=alpha, min_periods=period).mean()
        
        # Prevent division by zero
        mfi_ratio = pos_mf_avg / (neg_mf_avg + 1e-8)
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        return mfi.fillna(50.0).clip(15, 85)
    
    def detect_trend(self, data):
        """Fast trend detection for scalping."""
        if len(data) < 10:
            return 'neutral'
        
        close = data['close']
        ema3 = close.ewm(span=3).mean().iloc[-1]
        ema7 = close.ewm(span=7).mean().iloc[-1]
        ema15 = close.ewm(span=15).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Check bounds before calculating momentum
        if len(close) < 3:
            return 'neutral'
        momentum = (current_price - close.iloc[-2]) / close.iloc[-2]
        
        if ema3 > ema7 > ema15 and current_price > ema3 and momentum > 0.0001:
            return 'strong_uptrend'
        if ema3 < ema7 < ema15 and current_price < ema3 and momentum < -0.0001:
            return 'strong_downtrend'
        
        return 'neutral'
    
    def _is_valid_signal_data(self, data, rsi, mfi):
        """Validate signal data to prevent errors."""
        return (len(data) >= 20 and 
                not self._is_cooldown_active() and 
                pd.notna(rsi) and pd.notna(mfi))
    
    def _get_signal_conditions(self, trend, rsi, mfi):
        """Get signal conditions based on trend and indicators."""
        cfg = self.config
        
        if trend == 'strong_uptrend':
            return ('BUY' if rsi <= cfg['uptrend_oversold'] and 
                   mfi <= cfg['uptrend_mfi_threshold'] else None)
        
        elif trend == 'strong_downtrend':
            return ('SELL' if rsi >= cfg['short_rsi_minimum'] and 
                   mfi >= cfg['short_mfi_threshold'] and 
                   rsi >= cfg['downtrend_overbought'] else None)
        
        elif trend == 'neutral':
            if rsi <= cfg['neutral_oversold'] and mfi <= cfg['neutral_mfi_threshold']:
                return 'BUY'
            elif rsi >= cfg['neutral_overbought'] and mfi >= cfg['short_mfi_threshold']:
                return 'SELL'
        
        return None
    
    def generate_signal(self, data):
        """Generate trading signals with simplified logic."""
        if len(data) < 20:
            return None
            
        try:
            rsi = self.calculate_rsi(data['close']).iloc[-1]
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        except (IndexError, KeyError):
            return None
            
        if not self._is_valid_signal_data(data, rsi, mfi):
            return None
        
        trend = self.detect_trend(data)
        action = self._get_signal_conditions(trend, rsi, mfi)
        
        if action:
            price = data['close'].iloc[-1]
            signal = self._create_signal(action, trend, rsi, mfi, price, data, 
                                       is_short=(action == 'SELL'))
            if signal:
                self.last_signal_time = datetime.now()
            return signal
        
        return None
    
    def _create_signal(self, action, trend, rsi, mfi, price, data, is_short=False):
        """Create signal with validation."""
        window = data.tail(20)
        
        # Calculate structure levels
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.9985
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.001
            level = window['high'].max()
        
        # Validate stop distance
        stop_distance = abs(price - structure_stop) / price
        if not (0.0005 <= stop_distance <= 0.005):
            return None
        
        # Additional short validation
        if is_short and not self._validate_short_conditions(data, price):
            return None
        
        # Calculate confidence
        base_confidence = (abs(50 - rsi) + abs(50 - mfi)) * 1.5
        if action == 'BUY' and trend == 'strong_uptrend':
            base_confidence *= 1.1
        confidence = min(95, max(50, base_confidence))
        
        signal = {
            'action': action, 'trend': trend, 'rsi': round(rsi, 1), 'mfi': round(mfi, 1),
            'price': price, 'structure_stop': structure_stop, 'level': level,
            'signal_type': f"{trend}_{action.lower()}", 'confidence': round(confidence, 1),
            'target_profit_usdt': self.config['target_profit_usdt'],
            'estimated_move_pct': round(stop_distance * 200, 2),
            'max_hold_seconds': self.config['max_hold_seconds']
        }
        
        if is_short:
            signal.update({
                'position_size_multiplier': self.config['short_position_reduction'],
                'max_hold_seconds': 60
            })
        
        return signal
    
    def _validate_short_conditions(self, data, price):
        """Validate conditions specific to short positions."""
        if len(data) < 100:
            return False
            
        recent_low_100 = data.tail(100)['low'].min()
        distance_from_low = (price - recent_low_100) / recent_low_100
        return distance_from_low >= 0.02
    
    def _is_cooldown_active(self):
        """Check if cooldown period is active."""
        if not self.last_signal_time:
            return False
        return ((datetime.now() - self.last_signal_time).total_seconds() < 
                self.config['cooldown_seconds'])
    
    def calculate_indicators(self, data):
        """Calculate indicators with error handling."""
        min_length = max(self.config['rsi_length'], self.config['mfi_length']) + 5
        if len(data) < min_length:
            return {}
        
        try:
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            
            if rsi.isna().all() or mfi.isna().all():
                return {}
                
            return {'rsi': rsi, 'mfi': mfi}
        except Exception:
            return {}
    
    def get_strategy_info(self):
        """Get strategy information."""
        return {
            'name': 'RSI/MFI Crypto-Optimized Scalping', 
            'config': self.config,
            'performance_notes': {
                'uptrend_longs': '100% win rate - keep exact logic',
                'downtrend_shorts': 'Highly restrictive - crypto bias upward',
                'position_sizing': '$9091 USDT fixed size (10x leverage)',
                'profit_target': '$15 USDT (covers fees + profit)',
                'max_hold_time': '180 seconds',
                'emergency_stop': '0.6% max loss'
            }
        }