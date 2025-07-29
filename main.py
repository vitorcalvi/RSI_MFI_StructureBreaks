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

def display_info(engine, total_equity, current_price):
    """Display trading info"""
    wallet_balance = engine.get_wallet_balance_only()
    
    print(f"💰 Total Equity: ${total_equity:,.2f} | Wallet: ${wallet_balance:,.2f} | Symbol: {engine.symbol}")
    print(f"⚙️ Leverage: {engine.risk_manager.leverage}x | Position: {engine.risk_manager.max_position_size*100:.1f}% of wallet")
    
    # Risk per trade
    position_value = wallet_balance * engine.risk_manager.max_position_size
    max_loss = position_value * engine.risk_manager.stop_loss_pct
    risk_pct = (max_loss / wallet_balance) * 100
    
    print(f"\n🚨 RISK PER TRADE:")
    print(f"💸 Max Loss: ${max_loss:.2f} ({risk_pct:.1f}% of wallet)")
    print(f"📊 Position Value: ${position_value:.2f}")
    
    print(f"\n🔒 PROFIT MANAGEMENT:")
    print(f"🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% → Trailing stop")
    print(f"💰 Profit Protection: {engine.risk_manager.profit_protection_threshold:.1f}% → Close position")
    
    print(f"\n🔄 REVERSAL THRESHOLDS:")
    print(f"📈 Profit: +{engine.risk_manager.profit_reversal_threshold:.1f}%")
    print(f"📉 Loss: {engine.risk_manager.loss_reversal_threshold:.1f}%")
    
    print(f"\n🎮 STRATEGY:")
    print(f"📈 RSI: {engine.strategy.params['oversold_level']}/{engine.strategy.params['overbought_level']} (Length: {engine.strategy.params['rsi_length']})")
    print(f"🎯 Trend Filter: {'ON' if engine.strategy.params.get('require_trend', False) else 'OFF'}")
    print(f"⏱️ Cooldown: {engine.strategy.params['signal_cooldown']} periods")

async def main():
    print("🤖 ZORA Trading Bot - Streamlined")
    print("=" * 50)
    
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection failed")
            return
        
        # Get current data
        total_equity = engine.get_account_balance()
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 0.086
        
        display_info(engine, total_equity, current_price)
        
        # Trading mode
        mode = "TESTNET" if engine.demo_mode else "🚨 LIVE"
        print("=" * 50)
        print(f"🚀 {mode} TRADING")
        print("=" * 50)
        
        if not engine.demo_mode:
            print("🚨 LIVE TRADING - REAL MONEY AT RISK")
            print("=" * 50)
        
        await engine.notifier.bot_started(engine.symbol, total_equity)
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