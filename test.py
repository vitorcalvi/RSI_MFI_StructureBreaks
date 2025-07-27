#!/usr/bin/env python3
"""
Simple connection test for the trading bot
Run this first to verify everything works
"""

import os
import sys
import asyncio
import ccxt
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ccxt_connection():
    """Test basic CCXT connection to Bybit testnet"""
    print("🔗 Testing CCXT connection...")
    
    try:
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'sandbox': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True
            }
        })
        
        # Load markets
        markets = exchange.load_markets()
        print(f"✅ Connected to Bybit testnet - {len(markets)} markets loaded")
        
        # Test data fetch
        symbol = 'SOL/USDT'
        if symbol in markets:
            ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=10)
            print(f"✅ Fetched {len(ohlcv)} candles for {symbol}")
            
            # Show last price
            last_price = ohlcv[-1][4]  # Close price
            print(f"💰 {symbol} last price: ${last_price:.4f}")
        else:
            print(f"❌ {symbol} not found in markets")
            
        return True
        
    except Exception as e:
        print(f"❌ CCXT connection failed: {e}")
        return False

def test_strategy_import():
    """Test strategy module import"""
    print("\n📊 Testing strategy import...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
        
        strategy = RSIMFICloudStrategy()
        print(f"✅ Strategy loaded with params: {strategy.params}")
        
        # Test with dummy data
        dates = pd.date_range('2024-01-01', periods=50, freq='1min')
        dummy_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100.5] * 50,
            'volume': [1000] * 50
        }, index=dates)
        
        # Test indicator calculation
        result = strategy.calculate_indicators(dummy_data)
        print(f"✅ Indicators calculated - RSI: {result['rsi'].iloc[-1]:.2f}, MFI: {result['mfi'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy import failed: {e}")
        return False

def test_telegram():
    """Test Telegram notification"""
    print("\n📱 Testing Telegram...")
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if bot_token and chat_id:
        try:
            from telegram import Bot
            import asyncio
            
            async def telegram_test():
                bot = Bot(token=bot_token)
                
                # Test connection
                await bot.get_me()
                print("✅ Telegram bot token is valid")
                
                # Send test message
                await bot.send_message(chat_id=chat_id, text="🤖 Trading bot test message")
                print("✅ Test message sent to Telegram")
                
            # Run in new event loop to avoid conflicts
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram_test())
                loop.close()
            except:
                # Fallback - try with existing loop
                asyncio.run(telegram_test())
            
            return True
            
        except Exception as e:
            print(f"❌ Telegram test failed: {e}")
            return False
    else:
        print("⚠️  Telegram credentials not found in .env")
        return False

async def test_full_bot():
    """Test the full bot initialization"""
    print("\n🤖 Testing full bot initialization...")
    
    try:
        from core.trade_engine import TradeEngine
        
        engine = TradeEngine()
        print("✅ Trade engine created successfully")
        
        # Test data fetching
        symbol = 'SOL/USDT'
        df = await engine.fetch_ohlcv(symbol, timeframe='1m', limit=20)
        
        if df is not None and len(df) > 0:
            print(f"✅ Data fetched: {len(df)} candles")
            print(f"📊 Last price: ${df['close'].iloc[-1]:.4f}")
            
            # Test signal generation
            signal = engine.strategy.generate_signal(df)
            if signal:
                print(f"🎯 Signal generated: {signal}")
            else:
                print("📈 No signal (normal)")
                
            return True
        else:
            print("❌ No data fetched")
            return False
            
    except Exception as e:
        print(f"❌ Full bot test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Trading Bot Tests\n" + "="*50)
    
    tests = [
        test_ccxt_connection,
        test_strategy_import,
        test_telegram,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    # Run async test
    try:
        async_result = asyncio.run(test_full_bot())
        results.append(async_result)
    except Exception as e:
        print(f"❌ Async test crashed: {e}")
        results.append(False)
    
    print("\n" + "="*50)
    print("📋 TEST SUMMARY:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED - Bot ready to run!")
        print("💡 Run: python main.py")
    else:
        print("\n⚠️  Some tests failed - check errors above")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)