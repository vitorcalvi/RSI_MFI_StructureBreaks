import ccxt
import pandas as pd
import ta
from datetime import datetime

# Quick debug script
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
exchange.set_sandbox_mode(True)

# Fetch data
symbol = 'SOL/USDT'
ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=100)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print(f"Data points: {len(df)}")
print(f"Last close: ${df['close'].iloc[-1]:.4f}")
print(f"Volume range: {df['volume'].min():.2f} - {df['volume'].max():.2f}")

# Calculate RSI
rsi = ta.momentum.RSIIndicator(close=df['close'], window=7).rsi()
print(f"\nRSI values (last 5):")
print(rsi.tail())

# Calculate MFI
mfi = ta.volume.MFIIndicator(
    high=df['high'],
    low=df['low'], 
    close=df['close'],
    volume=df['volume'],
    window=7
).money_flow_index()
print(f"\nMFI values (last 5):")
print(mfi.tail())

# Check for issues
print(f"\nRSI NaN count: {rsi.isna().sum()}")
print(f"MFI NaN count: {mfi.isna().sum()}")
print(f"Zero volume bars: {(df['volume'] == 0).sum()}")