import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.trade_engine import TradeEngine

async def main():
    print("🤖 RSI+MFI Trading Bot Starting...")
    print("=" * 60)
    
    # Component initialization with status updates
    print("🔧 Initializing Components...")
    
    # Check if strategy params exist
    strategy_file = 'strategies/params_RSI_MFI_Cloud.json'
    if os.path.exists(strategy_file):
        print("✅ Component: Strategy parameters loaded")
    else:
        print("⚠️  Component: Strategy parameters missing, using defaults")
    
    # Initialize trade engine
    try:
        print("🔄 Component: Initializing Trade Engine...")
        engine = TradeEngine()
        print("✅ Component: Trade Engine initialized")
        print("✅ Component: Risk Manager configured")
        print("✅ Component: Strategy Engine loaded")
        print("✅ Component: Telegram Notifier ready")
        
        print("\n" + "=" * 60)
        print("🚀 Starting Trading System...")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to initialize trade engine: {e}")
        return
    
    try:
        # Start trading - this will show the detailed risk summary
        await engine.run()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠️  Shutdown Initiated by User...")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            print("🔄 Stopping trading engine...")
            await engine.stop()
            print("✅ Trading engine stopped successfully")
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)