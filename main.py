import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trade_engine import TradeEngine

async def main():
    print("🤖 RSI+MFI Trading Bot Starting...")
    
    # Check if strategy params exist
    strategy_file = 'strategies/params_RSI_MFI_Cloud.json'
    if not os.path.exists(strategy_file):
        print(f"⚠️  Warning: {strategy_file} not found, using default parameters")
    
    # Initialize trade engine
    try:
        engine = TradeEngine()
        print("✅ Trade engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize trade engine: {e}")
        return
    
    try:
        # Start trading
        await engine.run()
    except KeyboardInterrupt:
        print("\n⚠️  Stopping bot...")
    except Exception as e:
        print(f"❌ Runtime error: {e}")
    finally:
        try:
            await engine.stop()
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)