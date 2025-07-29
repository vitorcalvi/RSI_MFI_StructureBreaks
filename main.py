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

def display_accurate_info(engine, total_equity, current_price):
    """Display ACCURATE risk management info - NO MISLEADING DATA"""
    
    # Get wallet balance for position sizing calculations
    wallet_balance = engine.get_wallet_balance_only()
    
    # Get actual risk calculations using wallet balance for position sizing
    risk_summary = engine.risk_manager.get_risk_summary(wallet_balance, current_price)
    
    print(f"💰 Total Equity: ${total_equity:,.2f} | Wallet: ${wallet_balance:,.2f} | Symbol: {engine.symbol}")
    print(f"⚙️ Leverage: {engine.risk_manager.leverage}x | Position: {engine.risk_manager.max_position_size*100:.1f}% of wallet")
    
    print(f"\n🚨 ACTUAL RISK PER TRADE:")
    print(f"💸 Max Account Loss: ${risk_summary['max_account_loss']:.2f} ({risk_summary['actual_account_risk_pct']:.1f}% of wallet)")
    print(f"📊 Margin Used: ${risk_summary['margin_used']:.2f} ({(risk_summary['margin_used']/wallet_balance)*100:.1f}% of wallet)")
    print(f"🎯 Notional Value: ${risk_summary['notional_value']:.2f}")
    
    print(f"\n🔒 PROFIT MANAGEMENT (NO TAKE PROFIT - HOLD UNTIL SIGNAL):")
    print(f"🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% of total equity → Trailing stop")
    print(f"💰 Profit Protection: {engine.risk_manager.profit_protection_threshold:.1f}% of total equity → Close position")
    
    print(f"\n🔄 REVERSAL LOGIC:")
    print(f"📈 Profit Reversal: +{engine.risk_manager.profit_reversal_threshold:.1f}% of total equity")
    print(f"📉 Loss Reversal: {engine.risk_manager.loss_reversal_threshold:.1f}% of total equity")
    
    print(f"\n🎮 STRATEGY PARAMETERS:")
    print(f"📈 RSI: {engine.strategy.params['oversold_level']}/{engine.strategy.params['overbought_level']} (Length: {engine.strategy.params['rsi_length']})")
    print(f"🎯 Trend Filter: {'ENABLED' if engine.strategy.params.get('require_trend', False) else 'DISABLED'}")
    print(f"⏱️ Signal Cooldown: {engine.strategy.params['signal_cooldown']} periods")

async def main():
    print("🤖 ZORA Trading Bot - SAFE PARAMETERS")
    print("=" * 60)
    
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection failed")
            return
        
        # Get current data - use total equity for P&L calculations
        total_equity = engine.get_account_balance()  # This now returns totalEquity
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        current_price = float(ticker['result']['list'][0]['lastPrice']) if ticker.get('retCode') == 0 else 0.086
        
        # Display accurate information
        display_accurate_info(engine, total_equity, current_price)
        
        # Trading mode
        mode = "TESTNET" if engine.demo_mode else "🚨 LIVE"
        print("=" * 60)
        print(f"🚀 {mode} TRADING")
        print("=" * 60)
        
        # Safety warning for live trading
        if not engine.demo_mode:
            print("🚨 LIVE TRADING ACTIVE - REAL MONEY AT RISK")
            print("=" * 60)
        
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