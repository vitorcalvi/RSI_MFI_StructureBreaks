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
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI manually to avoid ta library issues"""
        prices = pd.Series(prices)
        deltas = prices.diff()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Use Wilder's smoothing (similar to EMA)
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # For subsequent values, use Wilder's smoothing
        for i in range(period, len(prices)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill NaN with neutral value
        rsi = rsi.replace([np.inf, -np.inf], 50)  # Replace inf with neutral
        
        return rsi
    
    def calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate MFI manually to avoid ta library issues"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        mf_sign = typical_price.diff()
        positive_mf = money_flow.where(mf_sign > 0, 0)
        negative_mf = money_flow.where(mf_sign < 0, 0)
        
        # Calculate money flow ratio
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        # Avoid division by zero
        mf_ratio = positive_mf_sum / negative_mf_sum.replace(0, 0.0001)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + mf_ratio))
        
        # Handle edge cases
        mfi = mfi.fillna(50)
        mfi = mfi.replace([np.inf, -np.inf], 50)
        
        return mfi
        
    def calculate_indicators(self, df):
        """Calculate RSI and MFI indicators"""
        df = df.copy()
        
        # Calculate RSI manually
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_length'])
        
        # Calculate MFI manually
        df['mfi'] = self.calculate_mfi(
            df['high'], df['low'], df['close'], df['volume'],
            self.params['mfi_length']
        )
        
        # Add RSI momentum
        df['rsi_change'] = df['rsi'].diff().fillna(0)
        
        return df
    
    def generate_signal(self, df):
        """Generate trading signal with more triggers"""
        if len(df) < max(self.params['rsi_length'], self.params['mfi_length']) + 10:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get current values
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        current_mfi = df['mfi'].iloc[-1]
        rsi_momentum = df['rsi_change'].iloc[-1]
        
        # Validate values
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None
            
        if current_rsi <= 0 or current_rsi >= 100 or current_mfi <= 0 or current_mfi >= 100:
            return None
        
        # Decrease cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
        
        # Multiple trigger conditions for testing
        
        # 1. Classic crossover
        buy_crossover = prev_rsi <= self.params['oversold_level'] and current_rsi > self.params['oversold_level']
        sell_crossover = prev_rsi >= self.params['overbought_level'] and current_rsi < self.params['overbought_level']
        
        # 2. RSI reversal (momentum based)
        buy_reversal = (current_rsi < self.params['oversold_level'] + 5 and 
                       rsi_momentum > 2 and 
                       current_mfi < 40)
        
        sell_reversal = (current_rsi > self.params['overbought_level'] - 5 and 
                        rsi_momentum < -2 and 
                        current_mfi > 60)
        
        # 3. RSI/MFI divergence
        buy_divergence = (current_rsi < 45 and current_mfi < 35)
        sell_divergence = (current_rsi > 55 and current_mfi > 65)
        
        # Combine conditions
        buy_signal = (buy_crossover or buy_reversal or buy_divergence) and self.signal_cooldown == 0
        sell_signal = (sell_crossover or sell_reversal or sell_divergence) and self.signal_cooldown == 0
        
        # Generate signal
        signal = None
        current_price = df['close'].iloc[-1]
        
        if buy_signal and self.last_signal != 'BUY':
            signal = {
                'action': 'BUY',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1]
            }
            self.last_signal = 'BUY'
            self.signal_cooldown = 3  # Wait 3 bars before next signal
            
        elif sell_signal and self.last_signal != 'SELL':
            signal = {
                'action': 'SELL',
                'price': current_price,
                'rsi': round(current_rsi, 2),
                'mfi': round(current_mfi, 2),
                'timestamp': df.index[-1]
            }
            self.last_signal = 'SELL'
            self.signal_cooldown = 3  # Wait 3 bars before next signal
        
        return signal