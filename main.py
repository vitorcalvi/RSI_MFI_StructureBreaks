#!/usr/bin/env python3
"""
High-Frequency Crypto Scalping Bot - Streamlined Version
RSI + MFI Strategy with Dynamic Symbol Configuration
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
        
    async def start(self):
        """Start the trading bot"""
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._display_startup_info()
        self.running = True
        print("⚡ Starting trading loop...")
        
        while self.running:
            try:
                await self.engine.run_cycle()
                await self._check_profit_target()
                await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(2)
        
        await self._shutdown()
    
    async def _check_profit_target(self):
        """Monitor and close position when profit target is reached"""
        if not self.engine.position:
            return
        
        current_pnl = float(self.engine.position.get('unrealisedPnl', 0))
        
        if current_pnl >= self.profit_target:
            print(f"🎯 Profit target reached! PnL: ${current_pnl:.2f}")
            await self.engine._close_position(f"profit_target_${self.profit_target:.0f}")
            
            await self.engine.notifier.send_message(
                f"🎯 <b>PROFIT TARGET HIT</b>\n\n"
                f"📊 <b>Symbol:</b> {self.engine.symbol}\n"
                f"💰 <b>PnL:</b> +${current_pnl:.2f}\n"
                f"🎯 <b>Target:</b> ${self.profit_target:.2f}\n"
                f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
    
    async def _display_startup_info(self):
        """Display bot startup information"""
        balance = await self.engine.get_account_balance()
        strategy_info = self.engine.strategy.get_strategy_info()
        risk_config = self.engine.risk_manager.config
        
        print("⚡" * 60)
        print(f"🚀 {self.engine.symbol} HIGH-FREQUENCY SCALPING BOT")
        print("⚡" * 60)
        print(f"📊 Strategy: {strategy_info['name']}")
        print(f"📈 RSI({strategy_info['config']['rsi_length']}) + MFI({strategy_info['config']['mfi_length']})")
        print(f"⏱️ Max Hold: {risk_config['max_position_time']}s")
        print(f"💰 Risk: {risk_config['fixed_risk_pct']*100}%")
        print(f"🎯 Reward: {risk_config['reward_ratio']}:1")
        print(f"💵 Balance: ${balance:,.2f}")
        print(f"🎯 PROFIT TARGET: ${self.profit_target:.2f}")
        print("-" * 60)
        
        await self.engine.notifier.send_bot_status("started", "HF Scalping Mode Active")
    
    async def _shutdown(self):
        """Shutdown the bot safely"""
        print("\n🛑 Initiating bot shutdown...")
        self.running = False
        
        if self.engine.position:
            print("⚡ Force closing position...")
            await self.engine._close_position("Bot shutdown")
        
        await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
        print("✅ Bot stopped safely")

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
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical bot error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()