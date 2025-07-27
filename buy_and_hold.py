#!/usr/bin/env python3
"""
SOL Scalping Strategy on Bybit Testnet
- Long Scalps: $181.50-$183 → $187-$188 / $189.50-$191 (RSI < 35, volume spike)
- Short Scalps: $189.50-$191 → $185-$186 / $181.50-$183 (RSI > 65)
- Breakouts: >$191 → $193-$195 (long), <$180 → $176-$172 (short)
- Risk: 1% per trade, 0.5% SL, 3% daily loss limit
- Timeframe: 1-minute
- Hours: 9-11 AM, 2-4 PM EST
"""

import os
import time
import logging
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator
import pandas as pd
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv(override=True)
API_KEY = os.getenv('TESTNET_BYBIT_API_KEY')
API_SECRET = os.getenv('TESTNET_BYBIT_API_SECRET')
if not API_KEY or not API_SECRET:
    raise ValueError("API credentials not set in .env file")

# Configuration
SYMBOL = 'SOLUSDT'
RISK_PERCENTAGE = 0.01  # 1% risk per trade
DAILY_LOSS_LIMIT = 0.03  # 3% daily loss cap
STOP_LOSS_PERCENTAGE = 0.005  # 0.5% stop loss
TRADING_HOURS = [(9, 11), (14, 16)]  # 9-11 AM, 2-4 PM EST
RSI_PERIOD = 14
VOLUME_SPIKE_FACTOR = 1.5  # 50% above 20-period average volume

# Strategy Price Levels
LONG_ENTRY_LOW = 181.50
LONG_ENTRY_HIGH = 183.00
LONG_TARGET_1 = 187.50  # Midpoint of $187-$188
LONG_TARGET_2 = 190.25  # Midpoint of $189.50-$191
RSI_LONG_ENTRY = 35

SHORT_ENTRY_LOW = 189.50
SHORT_ENTRY_HIGH = 191.00
SHORT_TARGET_1 = 185.50  # Midpoint of $185-$186
SHORT_TARGET_2 = 182.25  # Midpoint of $181.50-$183
RSI_SHORT_ENTRY = 65

BREAKOUT_LONG_LEVEL = 191.00
BREAKOUT_LONG_TARGET = 194.00  # Midpoint of $193-$195
BREAKOUT_SHORT_LEVEL = 180.00
BREAKOUT_SHORT_TARGET = 174.00  # Midpoint of $176-$172

# Initialize Bybit testnet connection
exchange = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)

# Set up logging
logging.basicConfig(
    filename='sol_scalping.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_trading_time():
    """Check if current time is within trading hours (EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(pytz.utc).astimezone(est)
    hour = now.hour
    return any(start <= hour < end for start, end in TRADING_HOURS)

def get_current_price():
    """Fetch the current price of SOLUSDT."""
    ticker = exchange.get_tickers(category='linear', symbol=SYMBOL)
    if ticker['retCode'] == 0:
        return float(ticker['result']['list'][0]['lastPrice'])
    raise Exception(f"Failed to fetch price: {ticker['retMsg']}")

def get_klines():
    """Fetch 1-minute candlestick data for indicators."""
    klines = exchange.get_kline(category='linear', symbol=SYMBOL, interval='1', limit=100)
    if klines['retCode'] == 0:
        df = pd.DataFrame(klines['result']['list'], 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    raise Exception(f"Failed to fetch klines: {klines['retMsg']}")

def calculate_rsi(df):
    """Calculate 14-period RSI."""
    rsi_indicator = RSIIndicator(df['close'], window=RSI_PERIOD)
    return rsi_indicator.rsi().iloc[-1]

def is_volume_spike(df):
    """Detect if current volume exceeds 1.5x the 20-period average."""
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    return current_volume > avg_volume * VOLUME_SPIKE_FACTOR

def get_balance():
    """Fetch USDT balance from the wallet."""
    resp = exchange.get_wallet_balance(accountType='UNIFIED')
    if resp['retCode'] == 0:
        for coin in resp['result']['list'][0]['coin']:
            if coin['coin'] == 'USDT':
                return float(coin['walletBalance'])
    raise Exception(f"Failed to fetch balance: {resp['retMsg']}")

def calculate_position_size(balance, current_price, stop_loss_price):
    """Calculate position size based on 1% risk."""
    risk_amount = balance * RISK_PERCENTAGE
    stop_loss_distance = abs(current_price - stop_loss_price) / current_price
    position_size_usdt = risk_amount / stop_loss_distance
    quantity = position_size_usdt / current_price
    return round(quantity, 3)  # Rounded to 3 decimals for SOL

def place_order(side, quantity):
    """Place a market order."""
    order = exchange.place_order(
        category='linear',
        symbol=SYMBOL,
        side=side,
        orderType='Market',
        qty=str(quantity)
    )
    if order['retCode'] == 0:
        return order['result']['orderId']
    raise Exception(f"Order failed: {order['retMsg']}")

def set_stop_loss_take_profit(side, current_price, quantity, target):
    """Set stop loss and take profit levels."""
    if side == 'Buy':
        stop_loss = current_price * (1 - STOP_LOSS_PERCENTAGE)
        take_profit = target
    else:  # Sell
        stop_loss = current_price * (1 + STOP_LOSS_PERCENTAGE)
        take_profit = target

    result = exchange.set_trading_stop(
        category='linear',
        symbol=SYMBOL,
        positionIdx=0,  # One-way mode
        stopLoss=str(round(stop_loss, 2)),
        takeProfit=str(round(take_profit, 2)),
        slTriggerBy='LastPrice',
        tpTriggerBy='LastPrice'
    )
    if result['retCode'] != 0:
        raise Exception(f"Failed to set stops: {result['retMsg']}")

def has_open_position():
    """Check if an open position exists."""
    pos = exchange.get_positions(category='linear', symbol=SYMBOL)
    if pos['retCode'] == 0 and pos['result']['list']:
        return float(pos['result']['list'][0]['size']) > 0
    return False

def main():
    """Main trading loop."""
    daily_loss = 0.0  # Simplified; enhance for actual loss tracking
    logging.info("Starting SOL scalping strategy on Bybit testnet")
    print("SOL Scalping Strategy Started - Check sol_scalping.log for details")

    while True:
       

        try:
            # Fetch market data
            current_price = get_current_price()
            df = get_klines()
            rsi = calculate_rsi(df)
            volume_spike = is_volume_spike(df)
            balance = get_balance()

            # Check daily loss limit (placeholder implementation)
            if daily_loss >= DAILY_LOSS_LIMIT * balance:
                logging.warning("Daily loss limit reached. Stopping.")
                print("Daily loss limit reached. Stopping.")
                break

            # Avoid multiple positions
            if has_open_position():
                time.sleep(5)
                continue

            # Long Scalp Trade
            if (LONG_ENTRY_LOW <= current_price <= LONG_ENTRY_HIGH and 
                rsi < RSI_LONG_ENTRY and volume_spike):
                stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE)
                quantity = calculate_position_size(balance, current_price, stop_loss_price)
                order_id = place_order('Buy', quantity)
                set_stop_loss_take_profit('Buy', current_price, quantity, LONG_TARGET_1)
                logging.info(f"Long scalp placed at {current_price}, Target: {LONG_TARGET_1}, SL: {stop_loss_price}")
                print(f"Long scalp placed at {current_price}")

            # Short Scalp Trade
            elif (SHORT_ENTRY_LOW <= current_price <= SHORT_ENTRY_HIGH and 
                  rsi > RSI_SHORT_ENTRY):
                stop_loss_price = current_price * (1 + STOP_LOSS_PERCENTAGE)
                quantity = calculate_position_size(balance, current_price, stop_loss_price)
                order_id = place_order('Sell', quantity)
                set_stop_loss_take_profit('Sell', current_price, quantity, SHORT_TARGET_1)
                logging.info(f"Short scalp placed at {current_price}, Target: {SHORT_TARGET_1}, SL: {stop_loss_price}")
                print(f"Short scalp placed at {current_price}")

            # Long Breakout Trade
            elif current_price > BREAKOUT_LONG_LEVEL and volume_spike:
                stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE)
                quantity = calculate_position_size(balance, current_price, stop_loss_price)
                order_id = place_order('Buy', quantity)
                set_stop_loss_take_profit('Buy', current_price, quantity, BREAKOUT_LONG_TARGET)
                logging.info(f"Long breakout placed at {current_price}, Target: {BREAKOUT_LONG_TARGET}, SL: {stop_loss_price}")
                print(f"Long breakout placed at {current_price}")

            # Short Breakout Trade
            elif current_price < BREAKOUT_SHORT_LEVEL and volume_spike:
                stop_loss_price = current_price * (1 + STOP_LOSS_PERCENTAGE)
                quantity = calculate_position_size(balance, current_price, stop_loss_price)
                order_id = place_order('Sell', quantity)
                set_stop_loss_take_profit('Sell', current_price, quantity, BREAKOUT_SHORT_TARGET)
                logging.info(f"Short breakout placed at {current_price}, Target: {BREAKOUT_SHORT_TARGET}, SL: {stop_loss_price}")
                print(f"Short breakout placed at {current_price}")

        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"Error occurred: {e}")
            time.sleep(60)  # Wait before retrying

        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    main()