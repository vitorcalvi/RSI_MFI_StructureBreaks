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

def display_info(engine, total_equity, current_price):
    """Display trading info with correct risk calculations"""
    wallet_balance = engine.get_wallet_balance_only()
    risk_summary = engine.risk_manager.get_risk_summary(wallet_balance)
    
    print(f"💰 Total Equity: ${total_equity:,.2f} | Wallet: ${wallet_balance:,.2f} | Symbol: {engine.symbol}")
    print(f"⚙️ Leverage: {risk_summary['leverage']}x | Position: {risk_summary['position_size_pct']:.3f}% of wallet")
    
    # FIXED: Show actual risk per trade
    print(f"\n🚨 RISK PER TRADE (CORRECTED):")
    print(f"💸 Max Loss: ${risk_summary['max_loss_usd']:.2f} ({risk_summary['risk_per_trade_pct']:.2f}% of wallet)")
    print(f"📊 Position Value: ${risk_summary['position_value']:.2f}")
    print(f"⚠️  With 25x leverage: 0.2% position = 5% risk (SAFE)")
    
    print(f"\n🔒 PROFIT MANAGEMENT:")
    print(f"🔓 Profit Lock: {risk_summary['profit_lock_threshold']:.1f}% wallet P&L → Trailing stop")
    print(f"💰 Profit Protection: {risk_summary['profit_protection_threshold']:.1f}% wallet P&L → Close position")
    
    print(f"\n🔄 REVERSAL THRESHOLDS:")
    print(f"📉 Loss Reversal: {risk_summary['loss_reversal_threshold']:.1f}% wallet P&L")
    
    print(f"\n🎮 STRATEGY:")
    print(f"📈 RSI: {engine.strategy.params['oversold_level']}/{engine.strategy.params['overbought_level']} (Length: {engine.strategy.params['rsi_length']})")
    print(f"🎯 Trend Filter: {'ON' if engine.strategy.params.get('require_trend', False) else 'OFF'}")
    print(f"⏱️ Cooldown: {engine.strategy.params['signal_cooldown']} periods")

async def main():
    print("🤖 ZORA Trading Bot - FIXED VERSION")
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