#!/usr/bin/env python3
"""
High-Frequency Crypto Scalping Bot - Bybit
RSI(5) + MFI(5) + Break & Retest Strategy
15 Second Max Hold | 0.5% Risk | 1.5:1 Reward
"""

import asyncio
import signal
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class HFScalpingBot:
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
    
    async def start(self):
        try:
            if not self.engine.connect():
                print("❌ Failed to connect to exchange")
                return
            
            await self.display_startup_info()
            self.running = True
            
            while self.running:
                try:
                    await self.engine.run_cycle()
                    await asyncio.sleep(0.5)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()
    
    async def display_startup_info(self):
        print("⚡" * 60)
        print("🚀 STARTING ETHUSDT HIGH-FREQUENCY SCALPING BOT")
        print("⚡" * 60)
        print("📊 Strategy: RSI(5) + MFI(5) + Break & Retest")
        print("⏱️  Max Hold: 15 seconds")
        print("💰 Risk: 0.5% per trade")
        print("🎯 Reward: 1.5:1 ratio")
        print("🔄 Polling: 500ms")
        print("🛑 Emergency Stop: 2%")
        print("📈 Profit Lock: 0.3%")
        print("🔄 Trailing: 0.5%")
        
        await self.engine.notifier.send_bot_status("started", "HF Scalping Mode Active")
        
        print("-" * 60)
        print("⚡ HIGH-FREQUENCY MODE ACTIVE")
        print("-" * 60)
    
    async def shutdown(self):
        print("\n🛑 Initiating HF bot shutdown...")
        self.running = False
        
        if self.engine.position:
            print("⚡ Force closing position...")
            await self.engine.close_position("HF Bot shutdown")
        
        await self.engine.notifier.send_bot_status("stopped", "HF Bot safely shutdown")
        print("✅ HF Scalping bot stopped safely")

def signal_handler(signum, frame):
    print(f"\n⚡ Received signal {signum} - Emergency shutdown...")
    raise KeyboardInterrupt

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    bot = HFScalpingBot()
    
    try:
        print("⚡ Initializing High-Frequency Scalping Bot...")
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 HF Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical HF bot error: {e}")
        try:
            asyncio.run(bot.engine.notifier.send_error_alert("Critical Error", str(e)))
        except:
            pass