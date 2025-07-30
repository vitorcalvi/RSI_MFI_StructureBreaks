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

def display_info(engine, wallet_balance, current_price):
    """Display simple trading info"""
    risk_summary = engine.risk_manager.get_risk_summary(wallet_balance, current_price)
    
    print(f"💰 Wallet Balance: ${wallet_balance:,.2f}")
    print(f"📊 Symbol: {risk_summary['symbol']}")
    print(f"💸 Fixed Risk: ${risk_summary['fixed_risk_usd']:.0f} per trade ({risk_summary['risk_pct']:.2f}% of wallet)")
    print(f"🛑 Stop Loss: {risk_summary['stop_loss_pct']:.1f}%")
    print(f"🎯 Take Profit: {risk_summary['take_profit_pct']:.1f}%")
    print(f"📈 Risk/Reward: 1:{risk_summary['risk_reward_ratio']:.1f}")
    print(f"🔒 Profit Lock: {engine.risk_manager.profit_lock_threshold}% position profit")

async def main():
    print("🤖 Simple Trading Bot - Fixed $100 Risk")
    print("=" * 50)
    
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection failed")
            return
        
        # Get current data
        wallet_balance = engine.get_wallet_balance()
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 0.086
        
        display_info(engine, wallet_balance, current_price)
        
        # Trading mode
        mode = "TESTNET" if engine.demo_mode else "🚨 LIVE"
        print("=" * 50)
        print(f"🚀 {mode} TRADING")
        print("=" * 50)
        
        await engine.notifier.bot_started(engine.symbol, wallet_balance)
        await engine.run()
        
    except KeyboardInterrupt:
        print("\n⚠️ Shutdown by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        if engine:
            try:
                await engine.stop()
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