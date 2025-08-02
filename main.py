#!/usr/bin/env python3
"""
High-Frequency Crypto Scalping Bot - Streamlined
RSI + MFI Strategy with Fixed $10,000 USDT Position Size
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
        self.profit_target = 20.0
        
    async def start(self):
        """Start the trading bot"""
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup()
        self.running = True
        
        while self.running:
            try:
                await self.engine.run_cycle()
                await self._check_profit_target()
                await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(2)
        
        await self._shutdown()
    
    async def _check_profit_target(self):
        """Check profit target"""
        if not self.engine.position:
            return
        
        pnl = float(self.engine.position.get('unrealisedPnl', 0))
        if pnl >= self.profit_target:
            print(f"🎯 Profit target reached! PnL: ${pnl:.2f}")
            await self.engine._close_position(f"profit_target_${self.profit_target:.0f}")
            
            await self.engine.notifier.send_message(
                f"🎯 <b>PROFIT TARGET HIT</b>\n\n"
                f"📊 Symbol: {self.engine.symbol}\n"
                f"💰 PnL: +${pnl:.2f}\n"
                f"🎯 Target: ${self.profit_target:.2f}\n"
                f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
            )
    
    async def _startup(self):
        """Display startup info"""
        balance = await self.engine.get_account_balance()
        config = self.engine.strategy.config
        risk = self.engine.risk_manager.config
        
        print("⚡" * 60)
        print(f"🚀 {self.engine.symbol} HIGH-FREQUENCY SCALPING BOT")
        print("⚡" * 60)
        print(f"📊 RSI({config['rsi_length']}) + MFI({config['mfi_length']}) | Max Hold: {risk['max_position_time']}s")
        print(f"💰 Position: ${risk['fixed_position_usdt']:,.0f} | Balance: ${balance:,.2f}")
        print(f"🎯 PROFIT TARGET: ${self.profit_target:.2f}")
        print("-" * 60)
        
        await self.engine.notifier.send_bot_status("started", "HF Scalping Mode Active - Fixed $10K")
    
    async def _shutdown(self):
        """Shutdown bot"""
        print("\n🛑 Shutting down...")
        self.running = False
        
        if self.engine.position:
            await self.engine._close_position("Bot shutdown")
        
        await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
        print("✅ Bot stopped")

def _signal_handler(signum, frame):
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    bot = HFScalpingBot()
    
    try:
        print("⚡ Starting High-Frequency Scalping Bot...")
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()