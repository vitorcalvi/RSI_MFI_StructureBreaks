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
    """Display streamlined startup info with safe formatting"""
    try:
        mode = "Testnet" if engine.demo_mode else "🔴 LIVE"
        symbol = engine.symbol.replace('/', '') if engine.symbol else 'ETHUSDT'
        
        # Safe balance formatting
        balance_str = f"{wallet_balance:,.0f}" if wallet_balance is not None else "0"
        
        # Safe risk formatting
        risk_amount = getattr(engine.risk_manager, 'fixed_risk_usd', 100)
        risk_str = f"{risk_amount:.0f}" if risk_amount is not None else "100"
        
        print(f"🚀 {symbol} Bot Started | {mode} Mode | Balance: ${balance_str} | Risk: ${risk_str}/trade")
        
    except Exception as e:
        print(f"🚀 Bot Started | Mode: {'Testnet' if engine.demo_mode else 'Live'} | Status: Ready")

async def main():
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection Failed | Check API credentials")
            return
        
        # Get current data for startup display with safe handling
        try:
            wallet_balance = engine.get_wallet_balance()
            ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 3500.0
            
            display_startup_info(engine, wallet_balance, current_price)
            
            await engine.notifier.bot_started(engine.symbol, wallet_balance)
        except Exception as e:
            print(f"⚠️ Startup info error: {e}")
            print("🚀 Bot Starting | Ready to trade")
        
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