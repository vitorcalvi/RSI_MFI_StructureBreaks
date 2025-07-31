import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

class RSIMFICloudStrategy:
    REQUIRED_PARAMS = [
        'rsi_length', 'mfi_length',
        'oversold', 'overbought',
        'structure_lookback', 'structure_buffer_pct',
        'fixed_risk_pct', 'trailing_stop_pct', 'profit_lock_threshold', 'reward_ratio',
        'entry_fee_pct', 'exit_fee_pct'
    ]

    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self._load_config()
        missing = [p for p in self.REQUIRED_PARAMS if p not in self.params]
        if missing:
            raise ValueError(f"Missing required params: {', '.join(missing)}")
        self.last_signal = None
        self.recent_breaks = []
        self.retest_monitoring = None
        self.max_break_history = 3
        print("✅ Break & Retest pattern detection enabled")

    def _load_config(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
            with open(params_file, 'r') as f:
                self.params = json.load(f)
        except Exception as e:
            print(f"⚠️ Config load error: {e}")
            # Fallback params
            self.params = {
                'rsi_length': 5,
                'mfi_length': 3,
                'oversold': 45,
                'overbought': 55,
                'structure_lookback': 120,
                'structure_buffer_pct': 0.3,
                'fixed_risk_pct': 0.02,
                'trailing_stop_pct': 0.01,
                'profit_lock_threshold': 4.0,
                'reward_ratio': 5.0,
                'entry_fee_pct': 0.00055,
                'exit_fee_pct': 0.00055
            }

    @property
    def symbol(self):
        return getattr(self.risk_manager, 'symbol', 'ETH/USDT')

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        try:
            period = self.params.get('rsi_length', 5)
            if len(prices) < period + 1:
                return pd.Series(50.0, index=prices.index)
            
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            avg_gain = gains.ewm(span=period).mean()
            avg_loss = losses.ewm(span=period).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-4)
            rsi = 100 - 100 / (1 + rs)
            return rsi.fillna(50.0).clip(0, 100)
        except Exception:
            return pd.Series(50.0, index=prices.index)

    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        try:
            period = self.params.get('mfi_length', 3)
            if len(close) < period + 1:
                return pd.Series(50.0, index=close.index)
            
            tp = (high + low + close) / 3
            money_flow = tp * volume
            sign = tp.diff()
            pos_mf = money_flow.where(sign > 0, 0)
            neg_mf = money_flow.where(sign <= 0, 0)
            pos_ema = pos_mf.ewm(span=period).mean()
            neg_ema = neg_mf.ewm(span=period).mean()
            mf_ratio = pos_ema / neg_ema.replace(0, 1e-4)
            mfi = 100 - 100 / (1 + mf_ratio)
            return mfi.fillna(50.0).clip(0, 100)
        except Exception:
            return pd.Series(50.0, index=close.index)

    def get_structure_stop(self, df: pd.DataFrame, action: str, entry_price: float) -> float:
        try:
            lb = self.params.get('structure_lookback', 120)
            buf_pct = self.params.get('structure_buffer_pct', 0.3) / 100
            
            if len(df) < lb:
                buffer = entry_price * buf_pct
                if action == 'SELL':
                    return entry_price * 1.015 + buffer
                return entry_price * 0.985 - buffer
            
            recent = df.tail(lb)
            buffer = entry_price * buf_pct
            
            if action == 'SELL':
                high = recent['high'].max()
                return high + buffer
            
            low = recent['low'].min()
            return low - buffer
        except Exception:
            # Fallback stop
            if action == 'SELL':
                return entry_price * 1.02
            return entry_price * 0.98

    def detect_structure_break(self, df: pd.DataFrame, side: str, entry_price: float, current_price: float):
        try:
            lb = self.params.get('structure_lookback', 120)
            if len(df) < lb: 
                return None
            
            recent = df.tail(lb)
            buffer = entry_price * (self.params.get('structure_buffer_pct', 0.3) / 100)
            
            if side == 'sell':
                lvl = recent['high'].max()
                if current_price > lvl + buffer:
                    self._record_structure_break('resistance', lvl, current_price, df.index[-1])
                    return {'break_type': 'resistance_break', 'flip_signal': 'BUY'}
            else:
                lvl = recent['low'].min()
                if current_price < lvl - buffer:
                    self._record_structure_break('support', lvl, current_price, df.index[-1])
                    return {'break_type': 'support_break', 'flip_signal': 'SELL'}
            
            return None
        except Exception:
            return None

    def _record_structure_break(self, btype, level, price, time):
        try:
            rec = {'type': btype, 'level': level, 'price': price, 'time': time}
            self.recent_breaks.append(rec)
            if len(self.recent_breaks) > self.max_break_history:
                self.recent_breaks.pop(0)
            self.retest_monitoring = {'break': rec, 'since': time, 'bars': 15, 'found': False}
        except Exception:
            pass

    def _calculate_retest_strength(self, df: pd.DataFrame, lvl: float, direction: str) -> float:
        try:
            if len(df) < 10:
                return 0.6
            
            data = df.tail(10)
            
            # Volume factor
            vol_ratio = data['volume'].tail(3).mean() / max(data['volume'].head(3).mean(), 1.0)
            vol_score = 0.9 if 0.5 <= vol_ratio <= 0.8 else 0.7
            
            # Depth factor
            if direction == 'bullish':
                depth = (data['high'].max() - data['low'].min()) / max((data['high'].max() - lvl), 1.0)
            else:
                depth = (data['high'].max() - data['low'].min()) / max((lvl - data['low'].min()), 1.0)
            
            depth_score = 0.9 if 0.5 <= depth <= 0.8 else 0.7
            
            # Time factor
            bars = len(data)
            time_score = 0.8 if 5 <= bars <= 10 else 0.7
            
            return np.mean([vol_score, depth_score, time_score])
        except Exception:
            return 0.6

    def detect_break_and_retest(self, df: pd.DataFrame, current_price: float):
        try:
            if not self.retest_monitoring or len(df) < 10: 
                return None
            
            rec = self.retest_monitoring['break']
            bars = len(df) - len(df[df.index <= rec['time']])
            
            if bars > self.retest_monitoring['bars']:
                self.retest_monitoring = None
                return None
            
            low, high = rec['level'] * 0.98, rec['level'] * 1.02
            if not (low <= current_price <= high): 
                return None
            
            recent = df.tail(5)
            
            if rec['type'] == 'resistance':
                if recent['low'].min() >= rec['level'] * 0.998:
                    strength = self._calculate_retest_strength(df, rec['level'], 'bullish')
                    if strength >= 0.6:
                        stop = rec['level'] * 0.996
                        return {
                            'action': 'BUY',
                            'price': current_price,
                            'structure_stop': stop,
                            'signal_type': 'BREAK_RETEST',
                            'retest_strength': strength,
                            'rsi': 50.0,
                            'mfi': 50.0,
                            'timestamp': df.index[-1]
                        }
            else:
                if recent['high'].max() <= rec['level'] * 1.002:
                    strength = self._calculate_retest_strength(df, rec['level'], 'bearish')
                    if strength >= 0.6:
                        stop = rec['level'] * 1.004
                        return {
                            'action': 'SELL',
                            'price': current_price,
                            'structure_stop': stop,
                            'signal_type': 'BREAK_RETEST',
                            'retest_strength': strength,
                            'rsi': 50.0,
                            'mfi': 50.0,
                            'timestamp': df.index[-1]
                        }
            
            return None
        except Exception:
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            return df
        except Exception:
            # Return safe fallback
            df = df.copy()
            df['rsi'] = 50.0
            df['mfi'] = 50.0
            return df

    def generate_signal(self, df: pd.DataFrame):
        try:
            min_length = max(self.params.get('rsi_length', 5), self.params.get('mfi_length', 3)) + 5
            if len(df) < min_length:
                return None
            
            df = self.calculate_indicators(df)
            price = df['close'].iloc[-1]
            
            # Check retest first
            sig = self.detect_break_and_retest(df, price)
            if sig: 
                return sig
            
            # Safe indicator extraction
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
            mfi = df['mfi'].iloc[-1] if 'mfi' in df.columns else 50.0
            
            # Handle None/NaN values
            if rsi is None or pd.isna(rsi):
                rsi = 50.0
            if mfi is None or pd.isna(mfi):
                mfi = 50.0
            
            oversold = self.params.get('oversold', 45)
            overbought = self.params.get('overbought', 55)
            
            # BUY signal
            if (rsi < oversold and mfi < oversold and self.last_signal != 'BUY'):
                self.last_signal = 'BUY'
                stop = self.get_structure_stop(df, 'BUY', price)
                return {
                    'action': 'BUY',
                    'price': price,
                    'structure_stop': stop,
                    'signal_type': 'RSI_MFI',
                    'rsi': round(float(rsi), 2),
                    'mfi': round(float(mfi), 2),
                    'timestamp': df.index[-1]
                }
            
            # SELL signal
            if (rsi > overbought and mfi > overbought and self.last_signal != 'SELL'):
                self.last_signal = 'SELL'
                stop = self.get_structure_stop(df, 'SELL', price)
                return {
                    'action': 'SELL',
                    'price': price,
                    'structure_stop': stop,
                    'signal_type': 'RSI_MFI',
                    'rsi': round(float(rsi), 2),
                    'mfi': round(float(mfi), 2),
                    'timestamp': df.index[-1]
                }
            
            return None
        except Exception as e:
            # Silent fallback - no signals if error
            return None