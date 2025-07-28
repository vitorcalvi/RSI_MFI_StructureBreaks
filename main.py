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

def display_risk_summary(engine, balance, current_price):
    """Display risk management summary using centralized risk management"""
    print("\n📊 RISK MANAGEMENT SUMMARY")
    print("=" * 60)
    
    # Get comprehensive risk summary from centralized risk manager
    risk_summary = engine.risk_manager.get_risk_summary(balance, current_price)
    
    # Account info
    print(f"💰 Balance: ${risk_summary['balance']:,.2f} USDT")
    print(f"📊 Symbol: {engine.symbol}")
    print(f"⚡ Leverage: {risk_summary['leverage']}x")
    print(f"📈 Position Size: {engine.risk_manager.max_position_size*100:.1f}% per trade")
    print(f"🎯 Risk per Trade: {engine.risk_manager.risk_per_trade*100:.1f}%")
    
    print(f"\n📋 PRICE LEVELS @ ${current_price:.4f}:")
    print("-" * 40)
    
    # Long levels
    print("📈 LONG:")
    print(f"   🛑 Stop Loss: ${risk_summary['stop_loss_long']:.4f} ({engine.risk_manager.stop_loss_pct*100:.1f}% price move)")
    print(f"   🎯 Take Profit: ${risk_summary['take_profit_long']:.4f} ({engine.risk_manager.take_profit_pct*100:.1f}% price move)")
    print(f"   🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% account P&L")
    
    # Short levels
    print("\n📉 SHORT:")
    print(f"   🛑 Stop Loss: ${risk_summary['stop_loss_short']:.4f} ({engine.risk_manager.stop_loss_pct*100:.1f}% price move)")
    print(f"   🎯 Take Profit: ${risk_summary['take_profit_short']:.4f} ({engine.risk_manager.take_profit_pct*100:.1f}% price move)")
    print(f"   🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% account P&L")
    
    # Analysis
    print(f"\n⚖️ RISK ANALYSIS:")
    print("-" * 40)
    print(f"📊 Position Value: ${risk_summary['notional_value']:,.2f} USDT")
    print(f"💳 Margin Used: ${risk_summary['margin_used']:,.2f} USDT ({risk_summary['margin_pct']:.1f}%)")
    print(f"🎯 Risk/Reward: 1:{engine.risk_manager.take_profit_pct / engine.risk_manager.stop_loss_pct:.1f}")
    print(f"🔒 Trailing Distance: {risk_summary['trailing_distance_pct']:.1f}%")
    print(f"🔄 Loss Switch: {engine.risk_manager.loss_switch_threshold:.0f}% account P&L")
    print(f"💰 Profit Protection: {engine.risk_manager.profit_protection_threshold:.0f}% account P&L")
    
    # Strategy
    print(f"\n🎮 STRATEGY:")
    print("-" * 40)
    print(f"📈 RSI Length: {engine.strategy.params['rsi_length']}")
    print(f"💹 MFI Length: {engine.strategy.params['mfi_length']}")
    print(f"🔽 Oversold: {engine.strategy.params['oversold_level']}")
    print(f"🔼 Overbought: {engine.strategy.params['overbought_level']}")
    print(f"🎯 Trend Filter: {'ON' if engine.strategy.params.get('require_trend', False) else 'OFF'}")
    print(f"⏱️ Timeframe: {engine.timeframe}m")
    
    # Risk Thresholds Summary
    print(f"\n🚨 RISK THRESHOLDS:")
    print("-" * 40)
    print(f"🔓 Profit Lock: {engine.risk_manager.profit_lock_threshold:.1f}% account → Activate trailing stop")
    print(f"💰 Profit Protection: {engine.risk_manager.profit_protection_threshold:.1f}% account → Take profit & cooldown")
    print(f"🔄 Position Reversal: {engine.risk_manager.position_reversal_threshold:.1f}% account → Reverse on signal")
    print(f"🚨 Loss Switch: {engine.risk_manager.loss_switch_threshold:.1f}% account → Force reverse")
    print(f"⏸️ Cooldown: {engine.risk_manager.reversal_cooldown_cycles} cycles after profit protection")
    
    print("=" * 60)

async def main():
    print("🤖 ZORA Trading Bot")
    print("=" * 60)
    
    engine = None
    try:
        engine = TradeEngine()
        
        if not engine.connect():
            print("❌ Connection failed")
            return
        
        # Get data for display
        balance = engine.get_account_balance()
        ticker = engine.exchange.get_tickers(category="linear", symbol=engine.linear)
        
        if ticker.get('retCode') == 0:
            current_price = float(ticker['result']['list'][0]['lastPrice'])
        else:
            current_price = 0.086
        
        # Display summary using centralized risk management
        display_risk_summary(engine, balance, current_price)
        
        print(f"\n🚀 LIVE TRADING STARTED")
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