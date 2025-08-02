import pandas as pd
import numpy as np
import json
from datetime import datetime

class RSIMFIStrategy:
    def __init__(self, config_file="strategies/rsi_mfi.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.last_signal = None
        self.last_signal_time = None
        self.structure_levels = {'resistance': [], 'support': []}
        
        print("⚡ RSI/MFI Strategy initialized - HIGH FREQUENCY MODE")
        print(f"📊 RSI({self.config['rsi_length']}) + MFI({self.config['mfi_length']})")
        print(f"🚀 NO TREND FILTERING - Maximum signal frequency")
    
    def load_config(self):
        """Load strategy configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"❌ Strategy config load error: {e}")
            return {
                "rsi_length": 7, "mfi_length": 7, "trend_ema_length": 14,
                "uptrend_oversold": 50, "uptrend_overbought": 50,
                "downtrend_oversold": 50, "downtrend_overbought": 50,
                "structure_lookback": 50, "structure_buffer_pct": 0.001,
                "cooldown_seconds": 5, "min_profit_distance": 0.001
            }
    
    def calculate_rsi(self, prices, period=None):
        """Calculate RSI indicator"""
        try:
            if period is None:
                period = self.config['rsi_length']
            
            if len(prices) < period + 1:
                return pd.Series(50.0, index=prices.index)
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi = 100 - 100 / (1 + rs)
            
            return rsi.fillna(50.0).clip(0, 100)
        except Exception as e:
            print(f"❌ RSI calculation error: {e}")
            return pd.Series(50.0, index=prices.index)
    
    def calculate_mfi(self, high, low, close, volume, period=None):
        """Calculate MFI (Money Flow Index) indicator"""
        try:
            if period is None:
                period = self.config['mfi_length']
            
            if len(close) < period + 1:
                return pd.Series(50.0, index=close.index)
            
            tp = (high + low + close) / 3
            money_flow = tp * volume
            
            mf_sign = tp.diff()
            positive_mf = money_flow.where(mf_sign > 0, 0)
            negative_mf = money_flow.where(mf_sign <= 0, 0)
            
            positive_mf_sum = positive_mf.rolling(window=period).sum()
            negative_mf_sum = negative_mf.rolling(window=period).sum()
            
            mfi_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1e-10)
            mfi = 100 - 100 / (1 + mfi_ratio)
            
            return mfi.fillna(50.0).clip(0, 100)
        except Exception as e:
            print(f"❌ MFI calculation error: {e}")
            return pd.Series(50.0, index=close.index)
    
    def find_structure_levels(self, data):
        """Find key support and resistance levels"""
        try:
            lookback = self.config['structure_lookback']
            if len(data) < lookback:
                return {'support': [], 'resistance': []}
            
            recent_data = data.tail(lookback)
            highs = recent_data['high']
            lows = recent_data['low']
            
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(recent_data) - 2):
                current_high = highs.iloc[i]
                current_low = lows.iloc[i]
                
                if (current_high > highs.iloc[i-2] and current_high > highs.iloc[i-1] and
                    current_high > highs.iloc[i+1] and current_high > highs.iloc[i+2]):
                    resistance_levels.append(current_high)
                
                if (current_low < lows.iloc[i-2] and current_low < lows.iloc[i-1] and
                    current_low < lows.iloc[i+1] and current_low < lows.iloc[i+2]):
                    support_levels.append(current_low)
            
            return {
                'resistance': sorted(set(resistance_levels), reverse=True)[:3],
                'support': sorted(set(support_levels))[:3]
            }
            
        except Exception as e:
            print(f"❌ Structure levels error: {e}")
            return {'support': [], 'resistance': []}
    
    def calculate_indicators(self, data):
        """Calculate all required indicators"""
        try:
            if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 1:
                return {}
            
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            
            return {'rsi': rsi, 'mfi': mfi}
            
        except Exception as e:
            print(f"❌ Indicators calculation error: {e}")
            return {}
    
    def generate_signal(self, data):
        """Generate trading signal - HIGH FREQUENCY MODE (No trend filtering)"""
        try:
            if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 5:
                return None
            
            # Check cooldown
            if self.last_signal_time:
                time_since_signal = (datetime.now() - self.last_signal_time).total_seconds()
                if time_since_signal < self.config['cooldown_seconds']:
                    return None
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if not indicators:
                return None
            
            rsi = indicators['rsi']
            mfi = indicators['mfi']
            
            current_rsi = rsi.iloc[-1]
            current_mfi = mfi.iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # HIGH-FREQUENCY THRESHOLDS (Will trigger with your current values!)
            buy_threshold = 48   # RSI 46.9 ≤ 48 ✅ OR MFI 17.1 ≤ 48 ✅✅
            sell_threshold = 52  # RSI/MFI ≥ 52 for sells
            
            # Find structure levels
            structure = self.find_structure_levels(data)
            self.structure_levels = structure
            
            signal = None
            
            # HIGH-FREQUENCY SIGNAL GENERATION (No trend restrictions)
            if current_rsi <= buy_threshold or current_mfi <= buy_threshold:
                # BUY signal - either RSI OR MFI below threshold
                signal = self._create_buy_signal(current_price, current_rsi, current_mfi, structure, 
                                               {'oversold': buy_threshold, 'overbought': sell_threshold}, 'hf_mode')
                print(f"🎯 BUY TRIGGER: RSI:{current_rsi:.1f}≤{buy_threshold} OR MFI:{current_mfi:.1f}≤{buy_threshold}")
            
            elif current_rsi >= sell_threshold or current_mfi >= sell_threshold:
                # SELL signal - either RSI OR MFI above threshold  
                signal = self._create_sell_signal(current_price, current_rsi, current_mfi, structure,
                                                {'oversold': buy_threshold, 'overbought': sell_threshold}, 'hf_mode')
                print(f"🎯 SELL TRIGGER: RSI:{current_rsi:.1f}≥{sell_threshold} OR MFI:{current_mfi:.1f}≥{sell_threshold}")
            
            # Update signal tracking
            if signal:
                self.last_signal = signal
                self.last_signal_time = datetime.now()
            
            return signal
            
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            return None
    
    def _create_buy_signal(self, price, rsi, mfi, structure, thresholds, trend):
        """Helper to create BUY signal"""
        stop_loss = price * 0.999  # Tight 0.1% stop
        if structure['support']:
            nearest_support = max([s for s in structure['support'] if s < price], default=stop_loss)
            stop_loss = min(stop_loss, nearest_support * (1 - self.config['structure_buffer_pct']))
        
        return {
            'action': 'BUY',
            'signal_type': f'HighFrequency_BUY',
            'trend': trend,
            'rsi': rsi,
            'mfi': mfi,
            'thresholds': thresholds,
            'structure_stop': stop_loss,
            'level': structure['support'][0] if structure['support'] else price,
            'confidence': min(100, (thresholds['oversold'] - min(rsi, mfi)) * 5)
        }
    
    def _create_sell_signal(self, price, rsi, mfi, structure, thresholds, trend):
        """Helper to create SELL signal"""
        stop_loss = price * 1.001  # Tight 0.1% stop
        if structure['resistance']:
            nearest_resistance = min([r for r in structure['resistance'] if r > price], default=stop_loss)
            stop_loss = max(stop_loss, nearest_resistance * (1 + self.config['structure_buffer_pct']))
        
        return {
            'action': 'SELL',
            'signal_type': f'HighFrequency_SELL',
            'trend': trend,
            'rsi': rsi,
            'mfi': mfi,
            'thresholds': thresholds,
            'structure_stop': stop_loss,
            'level': structure['resistance'][0] if structure['resistance'] else price,
            'confidence': min(100, (min(rsi, mfi) - thresholds['overbought']) * 5)
        }
    
    def get_strategy_info(self):
        """Get current strategy configuration info"""
        return {
            'name': 'High-Frequency RSI/MFI Strategy',
            'config': self.config,
            'last_signal': self.last_signal,
            'structure_levels': self.structure_levels
        }
    
    def update_config(self, new_config):
        """Update strategy configuration"""
        try:
            self.config.update(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("✅ Strategy config updated")
        except Exception as e:
            print(f"❌ Strategy config update error: {e}")