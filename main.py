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

def display_risk_summary(risk_manager, strategy, balance, current_price, symbol, leverage, timeframe):
    """Display risk management summary"""
    print("\n📊 RISK MANAGEMENT SUMMARY")
    print("=" * 60)
    
    # Calculate levels
    stop_loss_long = current_price * (1 - risk_manager.stop_loss_pct)
    stop_loss_short = current_price * (1 + risk_manager.stop_loss_pct)
    take_profit_long = current_price * (1 + risk_manager.take_profit_pct)
    take_profit_short = current_price * (1 - risk_manager.take_profit_pct)
    break_even_long = current_price * (1 + risk_manager.break_even_pct)
    break_even_short = current_price * (1 - risk_manager.break_even_pct)
    
    account_risk = f"{risk_manager.stop_loss_pct * risk_manager.leverage * 100:.1f}%"
    account_reward = f"{risk_manager.take_profit_pct * risk_manager.leverage * 100:.1f}%"
    risk_reward_ratio = f"1:{risk_manager.take_profit_pct / risk_manager.stop_loss_pct:.1f}"
    
    # Account info
    print(f"💰 Balance: ${balance:,.2f} USDT")
    print(f"📊 Symbol: {symbol}")
    print(f"⚡ Leverage: {leverage}x")
    print(f"📈 Position Size: {risk_manager.max_position_size*100:.1f}% per trade")
    print(f"🎯 Risk per Trade: {risk_manager.risk_per_trade*100:.1f}%")
    
    print(f"\n📋 PRICE LEVELS @ ${current_price:.4f}:")
    print("-" * 40)
    
    # Long levels
    print("📈 LONG:")
    print(f"   🛑 Stop Loss: ${stop_loss_long:.4f} ({account_risk} risk)")
    print(f"   🎯 Take Profit: ${take_profit_long:.4f} ({account_reward} gain)")
    print(f"   🔓 Break Even: ${break_even_long:.4f} (lock trigger)")
    
    # Short levels
    print("\n📉 SHORT:")
    print(f"   🛑 Stop Loss: ${stop_loss_short:.4f} ({account_risk} risk)")
    print(f"   🎯 Take Profit: ${take_profit_short:.4f} ({account_reward} gain)")
    print(f"   🔓 Break Even: ${break_even_short:.4f} (lock trigger)")
    
    # Analysis
    print(f"\n⚖️ ANALYSIS:")
    print("-" * 40)
    print(f"📊 Risk/Reward: {risk_reward_ratio}")
    print(f"🎯 Win Rate Needed: {100 / (1 + (risk_manager.take_profit_pct / risk_manager.stop_loss_pct)):.0f}%")
    print(f"🔒 Trailing Distance: {risk_manager.trailing_stop_distance*100:.1f}%")
    print(f"🔄 Loss Switch: {abs(risk_manager.loss_switch_threshold)*100:.0f}%")
    
    # Strategy
    print(f"\n🎮 STRATEGY:")
    print("-" * 40)
    print(f"📈 RSI Length: {strategy.params['rsi_length']}")
    print(f"💹 MFI Length: {strategy.params['mfi_length']}")
    print(f"🔽 Oversold: {strategy.params['oversold_level']}")
    print(f"🔼 Overbought: {strategy.params['overbought_level']}")
    print(f"🎯 Trend Filter: {'ON' if strategy.params.get('require_trend', False) else 'OFF'}")
    print(f"⏱️ Timeframe: {timeframe}m")
    
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
        
        # Display summary
        display_risk_summary(
            engine.risk_manager,
            engine.strategy,
            balance,
            current_price,
            engine.symbol,
            engine.risk_manager.leverage,
            engine.timeframe
        )
        
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