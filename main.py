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
    print("🤖 ZORA Trading Bot Starting...")
    print("=" * 60)
    
    try:
        # Initialize and start trading
        engine = TradeEngine()
        await engine.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Shutdown by user...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        try:
            await engine.stop()
            print("✅ Bot stopped")
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Done")
    except Exception as e:
        print(f"❌ Fatal: {e}")
        sys.exit(1)