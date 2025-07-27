import os
import json
import asyncio
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

async def main():
    # Check if strategy params exist
    strategy_file = 'strategies/params_RSI_MFI_Cloud.json'
    if not os.path.exists(strategy_file):
        print(f"Error: {strategy_file} not found")
        return
    
    # Initialize trade engine
    engine = TradeEngine()
    
    try:
        # Start trading
        await engine.run()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        await engine.stop()
    except Exception as e:
        print(f"Error: {e}")
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())