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
            try:
                with open(params_file, 'r') as f:
                    self.params = json.load(f)
            except:
                # Default production params
                self.params = {
                    "rsi_length": 14,
                    "mfi_length": 14,
                    "oversold_level": 30,
                    "overbought_level": 70,
                    "require_volume": False,
                    "require_trend": False
                }
        
        self.last_signal = None
        self.signal_cooldown = 0
        
    def calculate_rsi(self, prices, period=14):
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
    
    def calculate_mfi(self, high, low, close, volume, period=14):
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
                print("⚠️  Zero volume detected - using price-based volume estimation")
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
        
        # Add trend indicators
        df['price_change'] = df['close'].pct_change()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_signal(self, df):
        """Generate production trading signals"""
        min_bars = max(self.params['rsi_length'], self.params['mfi_length']) + 5
        if len(df) < min_bars:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current and previous values
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
        current_mfi = df['mfi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Validate values
        if (pd.isna(current_rsi) or pd.isna(current_mfi) or 
            pd.isna(prev_rsi) or pd.isna(current_price)):
            return None
        
        # Decrease cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None
        
        # Production signal conditions
        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']
        
        # More conservative signals for live trading
        
        # Strong BUY: RSI oversold, MFI confirms, and momentum turning up
        buy_conditions = [
            current_rsi < oversold,  # RSI oversold
            current_mfi < 40,        # MFI also low
            current_rsi > prev_rsi,  # RSI turning up
            self.last_signal != 'BUY'
        ]
        
        # Strong SELL: RSI overbought, MFI confirms, and momentum turning down  
        sell_conditions = [
            current_rsi > overbought,  # RSI overbought
            current_mfi > 60,          # MFI also high
            current_rsi < prev_rsi,    # RSI turning down
            self.last_signal != 'SELL'
        ]
        
        # Additional volume confirmation (if available)
        if len(df) >= 20:
            volume_ratio = df['volume_ratio'].iloc[-1]
            if not pd.isna(volume_ratio):
                buy_conditions.append(volume_ratio > 1.2)  # Above average volume
                sell_conditions.append(volume_ratio > 1.2)
        
        # Generate signals
        if all(buy_conditions):
            self.last_signal = 'BUY'
            self.signal_cooldown = 10  # Longer cooldown for live trading
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1],
                'confidence': 'HIGH'
            }
            
        elif all(sell_conditions):
            self.last_signal = 'SELL'
            self.signal_cooldown = 10  # Longer cooldown for live trading
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1],
                'confidence': 'HIGH'
            }
        
        return None