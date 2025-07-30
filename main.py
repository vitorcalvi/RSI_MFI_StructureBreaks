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
    """FIXED: Display with correct leverage calculations - NO COOLDOWN"""
    wallet_balance = engine.get_wallet_balance_only()
    risk_summary = engine.risk_manager.get_risk_summary(wallet_balance)
    
    print(f"💰 Total Equity: ${total_equity:,.2f} | Wallet: ${wallet_balance:,.2f} | Symbol: {engine.symbol}")
    print(f"⚙️ Leverage: {risk_summary['leverage']}x | Position: {risk_summary['position_size_pct']:.3f}% of wallet")
    
    # FIXED: Show actual risk per trade
    print(f"\n🚨 RISK PER TRADE (FIXED):")
    print(f"💸 Max Loss: ${risk_summary['max_loss_usd']:.2f} ({risk_summary['risk_per_trade_pct']:.3f}% of wallet)")
    print(f"📊 Position Value: ${risk_summary['position_value']:.2f}")
    print(f"⚠️  Stop Loss creates {risk_summary['risk_per_trade_pct']:.3f}% wallet risk (CORRECT)")
    
    print(f"\n🔒 PROFIT MANAGEMENT (FIXED - Position P&L Thresholds):")
    print(f"🔓 Profit Lock: {risk_summary['profit_lock_threshold']:.1f}% position P&L → {risk_summary['wallet_profit_lock']:.2f}% wallet impact")
    print(f"💰 Profit Protection: {risk_summary['profit_protection_threshold']:.1f}% position P&L → {risk_summary['wallet_profit_protection']:.2f}% wallet impact")
    
    print(f"\n🔄 REVERSAL THRESHOLDS (FIXED - NO COOLDOWN):")
    print(f"📉 Loss Reversal: -10.0% position P&L → -1.00% wallet impact")
    
    print(f"\n🎮 STRATEGY:")
    print(f"📈 RSI: {engine.strategy.params['oversold_level']}/{engine.strategy.params['overbought_level']} (Length: {engine.strategy.params['rsi_length']})")
    print(f"🎯 Trend Filter: {'ON' if engine.strategy.params.get('require_trend', False) else 'OFF'}")
    print(f"⏱️ Cooldown: COMPLETELY REMOVED")
    
    print(f"\n✅ LEVERAGE INTEGRATION:")
    print(f"📊 25x leverage PROPERLY calculated in all risk thresholds")
    print(f"🎯 Position P&L% used for triggers (accounts for leverage automatically)")
    print(f"💡 Wallet impact shown for transparency")
    print(f"🚀 NO COOLDOWN - Instant reversals when conditions met")

async def main():
    print("🤖 ZORA Trading Bot - COOLDOWN REMOVED & REVERSAL FIXED")
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