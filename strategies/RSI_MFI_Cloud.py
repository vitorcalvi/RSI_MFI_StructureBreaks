import pandas as pd
import numpy as np
from datetime import datetime

class RSIMFICloudStrategy:
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.params = risk_manager.config
        self.last_signal = None
        self.last_signal_time = None
        self.structure_levels = {'resistance': [], 'support': []}
        
        print("⚡ HF RSI/MFI Scalping strategy initialized")
        print(f"📊 RSI({self.params['rsi_length']}) + MFI({self.params['mfi_length']})")
    
    def calculate_rsi(self, prices, period=None):
        try:
            if period is None:
                period = self.params['rsi_length']
            
            if len(prices) < period + 1:
                return pd.Series(50.0, index=prices.index)
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(span=period).mean()
            avg_loss = loss.ewm(span=period).mean()
            
            rs = avg_gain / avg_loss.replace(0, 1e-4)
            rsi = 100 - 100 / (1 + rs)
            
            return rsi.fillna(50.0).clip(0, 100)
        except Exception as e:
            print(f"❌ RSI calculation error: {e}")
            return pd.Series(50.0, index=prices.index)
    
    def calculate_mfi(self, high, low, close, volume, period=None):
        try:
            if period is None:
                period = self.params['mfi_length']
            
            if len(close) < period + 1:
                return pd.Series(50.0, index=close.index)
            
            tp = (high + low + close) / 3
            money_flow = tp * volume
            
            mf_sign = tp.diff()
            positive_mf = money_flow.where(mf_sign > 0, 0)
            negative_mf = money_flow.where(mf_sign <= 0, 0)
            
            positive_ema = positive_mf.ewm(span=period).mean()
            negative_ema = negative_mf.ewm(span=period).mean()
            
            mfi_ratio = positive_ema / negative_ema.replace(0, 1e-4)
            mfi = 100 - 100 / (1 + mfi_ratio)
            
            return mfi.fillna(50.0).clip(0, 100)
        except Exception as e:
            print(f"❌ MFI calculation error: {e}")
            return pd.Series(50.0, index=close.index)
    
    def calculate_indicators(self, df):
        try:
            df = df.copy()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            self.update_structure_levels(df)
            return df
        except Exception as e:
            print(f"❌ Indicator calculation error: {e}")
            df['rsi'] = 50.0
            df['mfi'] = 50.0
            return df
    
    def update_structure_levels(self, df):
        try:
            lookback = self.params['structure_lookback']
            if len(df) < lookback:
                return
            
            recent_data = df.tail(lookback)
            self.structure_levels['resistance'] = self.find_pivot_highs(recent_data)
            self.structure_levels['support'] = self.find_pivot_lows(recent_data)
        except Exception as e:
            print(f"❌ Structure update error: {e}")
    
    def find_pivot_highs(self, df, window=3):
        try:
            highs = []
            for i in range(window, len(df) - window):
                current_high = df['high'].iloc[i]
                is_pivot = True
                
                for j in range(max(0, i - window), min(len(df), i + window + 1)):
                    if j != i and df['high'].iloc[j] >= current_high:
                        is_pivot = False
                        break
                
                if is_pivot:
                    highs.append(current_high)
            
            return sorted(highs, reverse=True)[:3]
        except Exception:
            return []
    
    def find_pivot_lows(self, df, window=3):
        try:
            lows = []
            for i in range(window, len(df) - window):
                current_low = df['low'].iloc[i]
                is_pivot = True
                
                for j in range(max(0, i - window), min(len(df), i + window + 1)):
                    if j != i and df['low'].iloc[j] <= current_low:
                        is_pivot = False
                        break
                
                if is_pivot:
                    lows.append(current_low)
            
            return sorted(lows)[:3]
        except Exception:
            return []
    
    def get_structure_stop(self, df, action, entry_price):
        try:
            buffer_pct = self.params['structure_buffer_pct'] / 100
            
            if action == 'BUY':
                support_levels = self.structure_levels.get('support', [])
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < entry_price], default=None)
                    if nearest_support:
                        return nearest_support - (entry_price * buffer_pct)
                
                if len(df) >= 10:
                    recent_low = df['low'].tail(10).min()
                    return recent_low - (entry_price * buffer_pct)
                
                return entry_price * 0.995
            
            else:
                resistance_levels = self.structure_levels.get('resistance', [])
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=None)
                    if nearest_resistance:
                        return nearest_resistance + (entry_price * buffer_pct)
                
                if len(df) >= 10:
                    recent_high = df['high'].tail(10).max()
                    return recent_high + (entry_price * buffer_pct)
                
                return entry_price * 1.005
        except Exception as e:
            print(f"❌ Structure stop error: {e}")
            if action == 'BUY':
                return entry_price * 0.99
            else:
                return entry_price * 1.01
    
    def detect_break_and_retest(self, df, current_price):
        try:
            if len(df) < 20:
                return None
            
            buffer_pct = self.params['structure_buffer_pct'] / 100
            
            resistance_levels = self.structure_levels.get('resistance', [])
            for resistance in resistance_levels:
                if (current_price > resistance and 
                    current_price < resistance * (1 + buffer_pct * 2)):
                    
                    recent_prices = df['close'].tail(5)
                    if any(price <= resistance for price in recent_prices):
                        return {
                            'action': 'BUY',
                            'price': current_price,
                            'structure_stop': resistance - (current_price * buffer_pct),
                            'signal_type': 'Break_Retest',
                            'level': resistance,
                            'timestamp': df.index[-1]
                        }
            
            support_levels = self.structure_levels.get('support', [])
            for support in support_levels:
                if (current_price < support and 
                    current_price > support * (1 - buffer_pct * 2)):
                    
                    recent_prices = df['close'].tail(5)
                    if any(price >= support for price in recent_prices):
                        return {
                            'action': 'SELL',
                            'price': current_price,
                            'structure_stop': support + (current_price * buffer_pct),
                            'signal_type': 'Break_Retest',
                            'level': support,
                            'timestamp': df.index[-1]
                        }
            
            return None
        except Exception as e:
            print(f"❌ Break & retest detection error: {e}")
            return None
    
    def generate_signal(self, df):
        try:
            min_length = max(self.params['rsi_length'], self.params['mfi_length']) + 5
            if len(df) < min_length:
                return None
            
            df = self.calculate_indicators(df)
            current_price = float(df['close'].iloc[-1])
            
            break_signal = self.detect_break_and_retest(df, current_price)
            if break_signal:
                self.last_signal = break_signal['action']
                self.last_signal_time = datetime.now()
                return break_signal
            
            rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0
            mfi = float(df['mfi'].iloc[-1]) if 'mfi' in df.columns else 50.0
            
            if pd.isna(rsi) or rsi is None:
                rsi = 50.0
            if pd.isna(mfi) or mfi is None:
                mfi = 50.0
            
            oversold = self.params['oversold']
            overbought = self.params['overbought']
            
            now = datetime.now()
            if (self.last_signal_time and 
                (now - self.last_signal_time).total_seconds() < self.params['cooldown_seconds']):
                return None
            
            if (rsi < oversold and mfi < oversold and self.last_signal != 'BUY'):
                stop_loss = self.get_structure_stop(df, 'BUY', current_price)
                
                signal = {
                    'action': 'BUY',
                    'price': current_price,
                    'structure_stop': stop_loss,
                    'signal_type': 'RSI_MFI_Oversold',
                    'rsi': round(rsi, 2),
                    'mfi': round(mfi, 2),
                    'timestamp': df.index[-1]
                }
                
                self.last_signal = 'BUY'
                self.last_signal_time = now
                return signal
            
            elif (rsi > overbought and mfi > overbought and self.last_signal != 'SELL'):
                stop_loss = self.get_structure_stop(df, 'SELL', current_price)
                
                signal = {
                    'action': 'SELL',
                    'price': current_price,
                    'structure_stop': stop_loss,
                    'signal_type': 'RSI_MFI_Overbought',
                    'rsi': round(rsi, 2),
                    'mfi': round(mfi, 2),
                    'timestamp': df.index[-1]
                }
                
                self.last_signal = 'SELL'
                self.last_signal_time = now
                return signal
            
            return None
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            return None
    
    def reset_signals(self):
        self.last_signal = None
        self.last_signal_time = None
        print("🔄 Signal tracking reset")