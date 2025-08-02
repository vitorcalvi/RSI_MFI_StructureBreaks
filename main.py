#!/usr/bin/env python3
"""
High-Frequency Crypto Scalping Bot - Bybit
RSI + MFI Strategy with Dynamic Symbol Configuration
Clean Architecture with Separation of Concerns + External PnL Management
"""

import asyncio
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class HFScalpingBot:
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        self.profit_target = 20.0  # $20 USD profit target
        self.last_pnl_check = 0
        
    async def start(self):
        """Start the trading bot"""
        try:
            if not self.engine.connect():
                print("❌ Failed to connect to exchange")
                return
            
            await self._display_startup_info()
            
            self.running = True
            print("⚡ Starting trading loop...")
            
            while self.running:
                try:
                    await self.engine.run_cycle()
                    
                    # External PnL monitoring (independent of trade engine)
                    await self._monitor_profit_target()
                    
                    await asyncio.sleep(0.5)  # 500ms cycle
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(2)
                    
        except KeyboardInterrupt:
            pass
        finally:
            await self._shutdown()
    
    async def _monitor_profit_target(self):
        """Monitor position PnL and close when profit target is reached"""
        try:
            if not self.engine.position:
                return
            
            current_pnl = float(self.engine.position.get('unrealisedPnl', 0))
            
            # Check if profit target reached
            if current_pnl >= self.profit_target:
                print(f"🎯 Profit target reached! PnL: ${current_pnl:.2f} >= ${self.profit_target:.2f}")
                print("⚡ Closing position due to profit target...")
                
                await self.engine._close_position(f"profit_target_${self.profit_target:.0f}")
                
                await self.engine.notifier.send_message(
                    f"🎯 <b>PROFIT TARGET HIT</b>\n\n"
                    f"📊 <b>Symbol:</b> {self.engine.symbol}\n"
                    f"💰 <b>PnL:</b> +${current_pnl:.2f}\n"
                    f"🎯 <b>Target:</b> ${self.profit_target:.2f}\n"
                    f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
                )
                return
            
            # Log PnL changes for monitoring
            if abs(current_pnl - self.last_pnl_check) > 0.5:  # Log when PnL changes by $0.50+
                self.last_pnl_check = current_pnl
                side = self.engine.position.get('side', '')
                size = self.engine.position.get('size', '0')
                entry_price = float(self.engine.position.get('avgPrice', 0))
                
                status = "📈 Position profitable" if current_pnl > 0 else "📉 Position underwater"
                pnl_display = f"+${current_pnl:.2f}" if current_pnl > 0 else f"${current_pnl:.2f}"
                print(f"{status}: {side} {size} @ ${entry_price:.2f} | PnL: {pnl_display}")
                
        except Exception as e:
            print(f"❌ Profit monitoring error: {e}")
    
    async def _display_startup_info(self):
        """Display bot startup information"""
        try:
            balance = await self.engine.get_account_balance()
            strategy_info = self.engine.strategy.get_strategy_info()
            risk_config = self.engine.risk_manager.config
            symbol = self.engine.symbol
            
            print("⚡" * 60)
            print(f"🚀 {symbol} HIGH-FREQUENCY SCALPING BOT")
            print("⚡" * 60)
            print(f"📊 Strategy: {strategy_info['name']}")
            print(f"📈 RSI({strategy_info['config']['rsi_length']}) + MFI({strategy_info['config']['mfi_length']})")
            print(f"⏱️ Max Hold: {risk_config['max_position_time']}s")
            print(f"💰 Risk: {risk_config['fixed_risk_pct']*100}%")
            print(f"🎯 Reward: {risk_config['reward_ratio']}:1")
            print(f"🔄 Polling: 500ms")
            print(f"💵 Balance: ${balance:,.2f}")
            print(f"🛑 Emergency Stop: {risk_config['emergency_stop_pct']*100}%")
            print(f"📈 Profit Lock: {risk_config['profit_lock_threshold']*100}%")
            print(f"🔄 Trailing: {risk_config['trailing_stop_pct']*100}%")
            print(f"🎯 EXTERNAL PROFIT TARGET: ${self.profit_target:.2f}")
            
            await self.engine.notifier.send_bot_status("started", "HF Scalping Mode Active")
            
            print("-" * 60)
            print("⚡ HIGH-FREQUENCY MODE ACTIVE")
            print(f"🎯 EXTERNAL PROFIT MONITORING: ${self.profit_target:.2f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Startup info error: {e}")
    
    async def _shutdown(self):
        """Shutdown the bot safely"""
        try:
            print("\n🛑 Initiating bot shutdown...")
            self.running = False
            
            if self.engine.position:
                print("⚡ Force closing position...")
                await self.engine._close_position("Bot shutdown")
            
            await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
            print("✅ Bot stopped safely")
            
        except Exception as e:
            print(f"❌ Shutdown error: {e}")

def _signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print(f"\n⚡ Received signal {signum} - Emergency shutdown...")
    raise KeyboardInterrupt

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    bot = HFScalpingBot()
    
    try:
        print("⚡ Initializing High-Frequency Scalping Bot...")
        print(f"🎯 External profit target: ${bot.profit_target:.2f}")
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical bot error: {e}")
        try:
            async def send_error():
                await bot.engine.notifier.send_error_alert("Critical Error", str(e))
            asyncio.run(send_error())
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()