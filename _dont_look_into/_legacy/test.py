import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

class HHLLTradingBot:
    def __init__(self, symbol='BTC/USDT', timeframe='5m', 
                 pivot_length=5, risk_percent=1.0):
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.pivot_length = pivot_length
        self.risk_percent = risk_percent
        
        # Initialize Bybit exchange for futures
        self.exchange = ccxt.bybit({
            'apiKey': os.getenv('TESTNET_BYBIT_API_KEY'),
            'secret': os.getenv('TESTNET_BYBIT_API_SECRET'),
            'sandbox': os.getenv('DEMO_MODE', 'true').lower() == 'true',
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # For perpetual futures
            }
        })
        
        # Trading state
        self.position = None  # 'long', 'short', None
        self.entry_price = 0
        self.stop_loss = 0
        self.trend = None  # 'up', 'down', None
        
        # Pivot tracking
        self.pivot_highs = []
        self.pivot_lows = []
        self.last_hh = None
        self.last_hl = None
        self.last_lh = None
        self.last_ll = None
        
    def get_ohlcv_data(self, limit=100):
        """Fetch OHLCV data from exchange"""
        try:
            # Fetch data for perpetual futures
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            if "API key is invalid" in str(e):
                self.logger.error("API key is invalid. Please check your Bybit API credentials.")
                self.logger.error("Make sure your API key has 'Read' and 'Trade' permissions enabled.")
                self.logger.error("For testnet, create API keys at: https://testnet.bybit.com/app/user/api-management")
            else:
                self.logger.error(f"Error fetching data: {e}")
            return None
    
    def find_pivot_high(self, highs, index):
        """Find pivot high at given index"""
        if index < self.pivot_length or index >= len(highs) - self.pivot_length:
            return False
            
        current_high = highs.iloc[index]
        
        # Check left side
        for i in range(index - self.pivot_length, index):
            if highs.iloc[i] >= current_high:
                return False
                
        # Check right side
        for i in range(index + 1, index + self.pivot_length + 1):
            if highs.iloc[i] >= current_high:
                return False
                
        return True
    
    def find_pivot_low(self, lows, index):
        """Find pivot low at given index"""
        if index < self.pivot_length or index >= len(lows) - self.pivot_length:
            return False
            
        current_low = lows.iloc[index]
        
        # Check left side
        for i in range(index - self.pivot_length, index):
            if lows.iloc[i] <= current_low:
                return False
                
        # Check right side
        for i in range(index + 1, index + self.pivot_length + 1):
            if lows.iloc[i] <= current_low:
                return False
                
        return True
    
    def detect_pivots(self, df):
        """Detect pivot highs and lows"""
        pivot_highs = []
        pivot_lows = []
        
        for i in range(len(df)):
            if self.find_pivot_high(df['high'], i):
                pivot_highs.append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'timestamp': df['timestamp'].iloc[i]
                })
                
            if self.find_pivot_low(df['low'], i):
                pivot_lows.append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'timestamp': df['timestamp'].iloc[i]
                })
        
        return pivot_highs, pivot_lows
    
    def classify_hhll(self, pivot_highs, pivot_lows):
        """Classify Higher High, Lower Low patterns"""
        if len(pivot_highs) >= 2:
            last_high = pivot_highs[-1]
            prev_high = pivot_highs[-2]
            
            if last_high['price'] > prev_high['price']:
                self.last_hh = last_high
                self.logger.info(f"Higher High detected at {last_high['price']}")
            else:
                self.last_lh = last_high
                self.logger.info(f"Lower High detected at {last_high['price']}")
        
        if len(pivot_lows) >= 2:
            last_low = pivot_lows[-1]
            prev_low = pivot_lows[-2]
            
            if last_low['price'] > prev_low['price']:
                self.last_hl = last_low
                self.logger.info(f"Higher Low detected at {last_low['price']}")
            else:
                self.last_ll = last_low
                self.logger.info(f"Lower Low detected at {last_low['price']}")
    
    def determine_trend(self):
        """Determine current trend based on HHLL patterns"""
        # Uptrend: HH + HL
        if (self.last_hh and self.last_hl and 
            self.last_hh['timestamp'] > self.last_lh['timestamp'] if self.last_lh else True and
            self.last_hl['timestamp'] > self.last_ll['timestamp'] if self.last_ll else True):
            return 'up'
        
        # Downtrend: LH + LL
        elif (self.last_lh and self.last_ll and
              self.last_lh['timestamp'] > self.last_hh['timestamp'] if self.last_hh else True and
              self.last_ll['timestamp'] > self.last_hl['timestamp'] if self.last_hl else True):
            return 'down'
        
        return None
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """Calculate position size based on risk management"""
        risk_amount = account_balance * (self.risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
            
        position_size = risk_amount / price_diff
        return position_size
    
    def place_long_order(self, current_price):
        """Place long order for futures"""
        try:
            # Set stop loss below recent pivot low
            if self.last_hl:
                stop_loss = self.last_hl['price'] * 0.98  # 2% below
            else:
                stop_loss = current_price * 0.95  # 5% stop loss
            
            # Get account balance for futures
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            
            # Calculate position size for futures (in USD value)
            risk_amount = usdt_balance * (self.risk_percent / 100)
            price_diff = abs(current_price - stop_loss)
            position_value = (risk_amount / price_diff) if price_diff > 0 else 0
            
            # Convert to contract size (for BTC/USDT perpetual, 1 contract = 1 USD)
            position_size = position_value / current_price
            
            # Minimum order size check
            min_size = 0.001  # Minimum for BTC perpetual
            position_size = max(position_size, min_size)
            
            if position_size > 0 and usdt_balance > 10:  # Minimum balance check
                # Place market long order for futures
                order = self.exchange.create_market_order(
                    self.symbol, 
                    'buy', 
                    position_size,
                    current_price,
                    params={'timeInForce': 'IOC'}
                )
                
                self.position = 'long'
                self.entry_price = current_price
                self.stop_loss = stop_loss
                
                self.logger.info(f"Long position opened at {current_price}, Size: {position_size}, SL: {stop_loss}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error placing long order: {e}")
            
        return False
    
    def place_short_order(self, current_price):
        """Place short order for futures"""
        try:
            # Set stop loss above recent pivot high
            if self.last_lh:
                stop_loss = self.last_lh['price'] * 1.02  # 2% above
            else:
                stop_loss = current_price * 1.05  # 5% stop loss
            
            # Get account balance for futures
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            
            # Calculate position size for futures
            risk_amount = usdt_balance * (self.risk_percent / 100)
            price_diff = abs(stop_loss - current_price)
            position_value = (risk_amount / price_diff) if price_diff > 0 else 0
            
            # Convert to contract size
            position_size = position_value / current_price
            
            # Minimum order size check
            min_size = 0.001
            position_size = max(position_size, min_size)
            
            if position_size > 0 and usdt_balance > 10:
                # Place market short order for futures
                order = self.exchange.create_market_order(
                    self.symbol, 
                    'sell', 
                    position_size,
                    current_price,
                    params={'timeInForce': 'IOC'}
                )
                
                self.position = 'short'
                self.entry_price = current_price
                self.stop_loss = stop_loss
                
                self.logger.info(f"Short position opened at {current_price}, Size: {position_size}, SL: {stop_loss}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error placing short order: {e}")
            
        return False
    
    def close_position(self):
        """Close current position for futures"""
        try:
            if self.position:
                # Get open positions for futures
                positions = self.exchange.fetch_positions([self.symbol])
                
                for pos in positions:
                    if pos['size'] > 0:
                        side = 'sell' if pos['side'] == 'long' else 'buy'
                        order = self.exchange.create_market_order(
                            self.symbol, 
                            side, 
                            pos['size'],
                            params={'reduceOnly': True}
                        )
                        break
            
            self.logger.info(f"Position closed: {self.position}")
            self.position = None
            self.entry_price = 0
            self.stop_loss = 0
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def check_exit_conditions(self, current_price):
        """Check if position should be closed"""
        if not self.position:
            return
            
        # Stop loss check
        if self.position == 'long' and current_price <= self.stop_loss:
            self.logger.info("Stop loss hit for long position")
            self.close_position()
            return
            
        if self.position == 'short' and current_price >= self.stop_loss:
            self.logger.info("Stop loss hit for short position")
            self.close_position()
            return
        
        # Trend reversal check
        current_trend = self.determine_trend()
        
        if self.position == 'long' and current_trend == 'down':
            self.logger.info("Trend reversal detected - closing long")
            self.close_position()
            
        elif self.position == 'short' and current_trend == 'up':
            self.logger.info("Trend reversal detected - closing short")
            self.close_position()
    
    def generate_signals(self, df):
        """Generate trading signals based on HHLL patterns"""
        current_price = df['close'].iloc[-1]
        
        # Check exit conditions first
        self.check_exit_conditions(current_price)
        
        if self.position:
            return  # Already in position
        
        # Detect pivots
        pivot_highs, pivot_lows = self.detect_pivots(df)
        
        if not pivot_highs or not pivot_lows:
            return
        
        # Update pivot lists
        self.pivot_highs = pivot_highs
        self.pivot_lows = pivot_lows
        
        # Classify patterns
        self.classify_hhll(pivot_highs, pivot_lows)
        
        # Determine trend
        current_trend = self.determine_trend()
        
        # Generate signals
        if current_trend == 'up' and not self.position:
            # Look for Higher Low formation to enter long
            if self.last_hl and self.last_hl['timestamp'] == df['timestamp'].iloc[-self.pivot_length-1]:
                self.logger.info("Higher Low detected - considering long entry")
                self.place_long_order(current_price)
                
        elif current_trend == 'down' and not self.position:
            # Look for Lower High formation to enter short
            if self.last_lh and self.last_lh['timestamp'] == df['timestamp'].iloc[-self.pivot_length-1]:
                self.logger.info("Lower High detected - considering short entry")
                self.place_short_order(current_price)
    
    def run(self):
        """Main trading loop"""
        self.logger.info(f"Starting HHLL Trading Bot for {self.symbol}")
        
        # Verify API credentials first
        try:
            self.exchange.load_markets()
            self.logger.info("Bybit API credentials verified successfully")
        except Exception as e:
            if "API key is invalid" in str(e):
                self.logger.error("API key is invalid. Please check your .env file:")
                self.logger.error("1. Go to https://testnet.bybit.com/app/user/api-management")
                self.logger.error("2. Create API key with Read + Trade + Derivatives permissions")
                self.logger.error("3. Update .env with full API key and secret")
                return
            else:
                self.logger.error(f"API connection error: {e}")
                return
        
        while True:
            try:
                # Fetch latest data
                df = self.get_ohlcv_data()
                
                if df is not None and len(df) > self.pivot_length * 2:
                    # Generate signals
                    self.generate_signals(df)
                    
                    # Log current status
                    current_price = df['close'].iloc[-1]
                    trend = self.determine_trend()
                    self.logger.info(f"Price: {current_price}, Trend: {trend}, Position: {self.position}")
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                if self.position:
                    self.close_position()
                break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(30)

# Usage
if __name__ == "__main__":
    # Initialize bot
    bot = HHLLTradingBot(
        symbol='BTC/USDT',
        timeframe='5m',
        pivot_length=5,
        risk_percent=1.0
    )
    
    # Start trading
    bot.run()