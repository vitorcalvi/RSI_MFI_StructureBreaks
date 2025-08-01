#!/usr/bin/env python3
"""
High-Frequency Crypto Scalping Bot - Bybit
RSI(5) + MFI(5) Strategy
Clean Architecture with Separation of Concerns
"""

import asyncio
import signal
import sys
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class HFScalpingBot:
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
    
    async def start(self):
        """Start the trading bot"""
        try:
            # Connect to exchange
            if not self.engine.connect():
                print("❌ Failed to connect to exchange")
                return
            
            # Display startup information
            await self.display_startup_info()
            
            # Start trading loop
            self.running = True
            print("⚡ Starting trading loop...")
            
            while self.running:
                try:
                    await self.engine.run_cycle()
                    await asyncio.sleep(0.5)  # 500ms cycle
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(2)  # Wait before retry
                    
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()
    
    async def display_startup_info(self):
        """Display bot startup information"""
        try:
            # Get current balance
            balance = await self.engine.get_account_balance()
            
            # Get strategy info
            strategy_info = self.engine.strategy.get_strategy_info()
            risk_config = self.engine.risk_manager.config
            
            print("⚡" * 60)
            print("🚀 ADAUSDT HIGH-FREQUENCY SCALPING BOT")
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
            
            # Send startup notification
            await self.engine.notifier.send_bot_status("started", "HF Scalping Mode Active")
            
            print("-" * 60)
            print("⚡ HIGH-FREQUENCY MODE ACTIVE")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Startup info error: {e}")
    
    async def shutdown(self):
        """Shutdown the bot safely"""
        try:
            print("\n🛑 Initiating bot shutdown...")
            self.running = False
            
            # Close any open positions
            if self.engine.position:
                print("⚡ Force closing position...")
                await self.engine.close_position("Bot shutdown")
            
            # Send shutdown notification
            await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
            
            print("✅ Bot stopped safely")
            
        except Exception as e:
            print(f"❌ Shutdown error: {e}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print(f"\n⚡ Received signal {signum} - Emergency shutdown...")
    raise KeyboardInterrupt

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start bot
    bot = HFScalpingBot()
    
    try:
        print("⚡ Initializing High-Frequency Scalping Bot...")
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical bot error: {e}")
        try:
            # Try to send error notification
            async def send_error():
                await bot.engine.notifier.send_error_alert("Critical Error", str(e))
            asyncio.run(send_error())
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()