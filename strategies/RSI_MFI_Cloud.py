import json
import os
import pandas as pd
import numpy as np

class RSIMFICloudStrategy:
    def __init__(self, params=None):
        if params is not None:
            self.params = params
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            params_file = os.path.join(current_dir, 'params_RSI_MFI_Cloud.json')
            with open(params_file, 'r') as f:
                self.params = json.load(f)
        
        self.last_signal = None
        self.signal_cooldown = 0
        
    def calculate_rsi(self, prices, period=7):
        """Calculate RSI with robust error handling"""
        try:
            if len(prices) < period + 1:
                return pd.Series([50] * len(prices), index=prices.index)
                
            prices = pd.Series(prices).astype(float)
            deltas = prices.diff()
            
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            # Use exponential moving average for smoother RSI
            avg_gain = gains.ewm(span=period, adjust=False).mean()
            avg_loss = losses.ewm(span=period, adjust=False).mean()
            
            # Calculate RSI
            rs = avg_gain / avg_loss.replace(0, 0.0001)
            rsi = 100 - (100 / (1 + rs))
            
            # Clean data
            rsi = rsi.fillna(50)
            rsi = rsi.replace([np.inf, -np.inf], 50)
            rsi = rsi.clip(0, 100)
            
            return rsi
            
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_mfi(self, high, low, close, volume, period=7):
        """Calculate MFI with enhanced volume handling"""
        try:
            if len(close) < period + 1:
                return pd.Series([50] * len(close), index=close.index)
                
            # Ensure all series are numeric
            high = pd.Series(high).astype(float)
            low = pd.Series(low).astype(float)
            close = pd.Series(close).astype(float)
            volume = pd.Series(volume).astype(float)
            
            # Handle zero or missing volume
            if volume.sum() == 0 or volume.isna().all():
                # Estimate volume based on price volatility
                price_range = high - low
                avg_price = (high + low + close) / 3
                volume = price_range * avg_price * 1000  # Synthetic volume
            
            # Replace zero volume with minimum non-zero value
            min_volume = volume[volume > 0].min() if (volume > 0).any() else 1
            volume = volume.replace(0, min_volume)
            volume = volume.fillna(min_volume)
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            # Calculate positive and negative money flow
            mf_sign = typical_price.diff()
            positive_mf = money_flow.where(mf_sign > 0, 0)
            negative_mf = money_flow.where(mf_sign <= 0, 0)
            
            # Use exponential moving average for smoother MFI
            positive_mf_ema = positive_mf.ewm(span=period, adjust=False).mean()
            negative_mf_ema = negative_mf.ewm(span=period, adjust=False).mean()
            
            # Calculate MFI
            mf_ratio = positive_mf_ema / negative_mf_ema.replace(0, 0.0001)
            mfi = 100 - (100 / (1 + mf_ratio))
            
            # Clean data
            mfi = mfi.fillna(50)
            mfi = mfi.replace([np.inf, -np.inf], 50)
            mfi = mfi.clip(0, 100)
            
            return mfi
            
        except Exception as e:
            print(f"MFI calculation error: {e}")
            return pd.Series([50] * len(close), index=close.index)
    
    def detect_trend(self, close):
        """ENHANCED 2-EMA trend detection for ZORA/USDT 5-minute timeframe"""
        try:
            # Simple 2-EMA system optimized for 5-minute ZORA trading
            ema_fast = close.ewm(span=12).mean()  # 12-period EMA (1 hour)
            ema_slow = close.ewm(span=26).mean()  # 26-period EMA (2+ hours)
            
            # Current EMA values
            fast_current = ema_fast.iloc[-1]
            slow_current = ema_slow.iloc[-1]
            
            # Calculate EMA separation (as % of price)  
            price_current = close.iloc[-1]
            ema_distance_pct = abs(fast_current - slow_current) / price_current * 100
            
            # Trend logic with minimum separation filter
            min_separation = 0.15  # 0.15% minimum EMA separation for clear trend
            
            if fast_current > slow_current and ema_distance_pct >= min_separation:
                return "UP"
            elif fast_current < slow_current and ema_distance_pct >= min_separation:
                return "DOWN"
            else:
                return "SIDEWAYS"  # EMAs too close = consolidation
                
        except Exception as e:
            print(f"Trend detection error: {e}")
            return "SIDEWAYS"
        
    def calculate_indicators(self, df):
        """Calculate RSI and MFI indicators"""
        df = df.copy()
        
        # Validate input data
        if len(df) < 2:
            return df
            
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_length'])
        
        # Calculate MFI with enhanced volume handling
        df['mfi'] = self.calculate_mfi(
            df['high'], df['low'], df['close'], df['volume'],
            self.params['mfi_length']
        )
        
        # Add enhanced 2-EMA trend detection
        df['trend'] = self.detect_trend(df['close'])
        
        # Add additional trend indicators
        df['price_change'] = df['close'].pct_change()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_signal(self, df):
        """Generate trading signals with ENHANCED TREND FILTER"""
        min_bars = max(self.params['rsi_length'], self.params['mfi_length'], 26) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current and previous values
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
        current_mfi = df['mfi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_trend = df['trend'].iloc[-1]
        
        # Validate values
        if (pd.isna(current_rsi) or pd.isna(current_mfi) or 
            pd.isna(prev_rsi) or pd.isna(current_price)):
            return None
        
        # Signal cooldown for real trading
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None
        
        # Trading signal conditions
        oversold = self.params['oversold_level']    # 35
        overbought = self.params['overbought_level'] # 65
        
        # ENHANCED: BUY only in clear UPTREND (not sideways)
        buy_conditions = [
            current_rsi < oversold,
            current_mfi < oversold,
            current_trend == "UP" if self.params['require_trend'] else True,  # TREND FILTER
            self.last_signal != 'BUY'
        ]
        
        # ENHANCED: SELL only in clear DOWNTREND (not sideways)
        sell_conditions = [
            current_rsi > overbought,
            current_mfi > overbought,
            current_trend == "DOWN" if self.params['require_trend'] else True,  # TREND FILTER
            self.last_signal != 'SELL'
        ]
        
        # Generate signals
        if all(buy_conditions):
            self.last_signal = 'BUY'
            self.signal_cooldown = 5  # 5 cycles cooldown for real trading
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1],
                'confidence': 'ENHANCED_TREND_FILTERED'
            }
            
        elif all(sell_conditions):
            self.last_signal = 'SELL'
            self.signal_cooldown = 5  # 5 cycles cooldown for real trading
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'trend': current_trend,
                'timestamp': df.index[-1],
                'confidence': 'ENHANCED_TREND_FILTERED'
            }
        
        return None