import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.trade_engine import TradeEngine

def display_essential_info(engine, balance, current_price):
    """Display essential risk management and strategy info"""
    print(f"💰 Balance: ${balance:,.2f} | Symbol: {engine.symbol} | Leverage: {engine.risk_manager.leverage}x")
    
    print(f"\n🚨 RISK MANAGEMENT:")
    print(f"🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% account → Trailing stop")
    print(f"💰 Profit Protection: {engine.risk_manager.profit_protection_threshold:.1f}% account → Close & cooldown")
    print(f"🔄 Position Reversal: {engine.risk_manager.position_reversal_threshold:.1f}% account → Reverse on signal")
    
    print(f"\n🎮 STRATEGY (Optimized for Higher Win Rate):")
    print(f"📈 RSI/MFI: {engine.strategy.params['oversold_level']}/{engine.strategy.params['overbought_level']} | Length: {engine.strategy.params['rsi_length']}")
    print(f"🎯 Trend Filter: {'ON' if engine.strategy.params.get('require_trend', False) else 'OFF'} | Position Size: {engine.risk_manager.max_position_size*100:.1f}%")

async def main():
    print("🤖 ZORA Trading Bot")
    print("=" * 60)
    
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection failed")
            return
        
        # Essential info only
        balance = engine.get_account_balance()
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 0.086
        
        display_essential_info(engine, balance, current_price)
        
        # Show trading mode clearly
        mode = "Testnet" if engine.demo_mode else "Live"
        print("=" * 60)
        print(f"🚀 {mode.upper()} TRADING")
        print("=" * 60)
        
        await engine.notifier.bot_started(engine.symbol, balance)
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