import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.trade_engine import TradeEngine

def display_startup_info(engine, wallet_balance, current_price):
    """Display streamlined startup info"""
    mode = "Testnet" if engine.demo_mode else "🔴 LIVE"
    symbol = engine.symbol.replace('/', '')
    
    print(f"🚀 {symbol} Bot Started | {mode} Mode | Balance: ${wallet_balance:,.0f} | Risk: ${engine.risk_manager.fixed_risk_usd:.0f}/trade")

async def main():
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection Failed | Check API credentials")
            return
        
        # Get current data for startup display
        wallet_balance = engine.get_wallet_balance()
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 0.086
        
        display_startup_info(engine, wallet_balance, current_price)
        
        
        await engine.notifier.bot_started(engine.symbol, wallet_balance)
        await engine.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown Initiated | Closing positions...")
        
    except Exception as e:
        print(f"\n❌ Fatal Error | {e}")
        
    finally:
        if engine:
            try:
                await engine.stop()
                print("✅ Bot Stopped | All positions closed | Session complete")
            except:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Done")
    except Exception as e:
        print(f"❌ Fatal: {e}")
        sys.exit(1)