import asyncio
import signal
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class HFScalpingBot:
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        
    async def start(self):
        """Start the trading bot"""
        if not self._validate_environment():
            return
            
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup()
        self.running = True
        
        while self.running:
            try:
                await self.engine.run_cycle()
                await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(2)
        
        await self._shutdown()
    
    def _validate_environment(self):
        """Validate environment configuration"""
        required_vars = [
            'TRADING_SYMBOL', 'DEMO_MODE', 
            'TESTNET_BYBIT_API_KEY', 'TESTNET_BYBIT_API_SECRET'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"❌ Missing environment variables: {', '.join(missing)}")
            print("📝 Check your .env file")
            return False
        
        return True
    
    async def _startup(self):
        """Display comprehensive startup info"""
        balance = await self.engine.get_account_balance()
        strategy_config = self.engine.strategy.config
        risk_config = self.engine.risk_manager.config
        strategy_info = self.engine.strategy.get_strategy_info()
        
        # Header
        print(f"\n🚀 {self.engine.symbol} HIGH-FREQUENCY SCALPING BOT")
        print("=" * 60)
        
        # Environment
        demo_mode = "TESTNET" if self.engine.demo_mode else "LIVE"
        print(f"🌐 Environment: {demo_mode}")
        print(f"💰 Account Balance: ${balance:,.2f} USDT")
        print(f"📊 Symbol: {self.engine.symbol}")
        
        # Strategy Configuration  
        print(f"\n⚙️  STRATEGY: {strategy_info['name']}")
        print("-" * 60)
        print(f"📈 RSI Length: {strategy_config['rsi_length']} | MFI Length: {strategy_config['mfi_length']}")
        print(f"🎯 Uptrend: RSI≤{strategy_config['uptrend_oversold']}, MFI≤{strategy_config['uptrend_mfi_threshold']}")
        print(f"📉 Neutral Long: RSI≤{strategy_config['neutral_oversold']}, MFI≤{strategy_config['neutral_mfi_threshold']}")
        print(f"⚡ Cooldown: {strategy_config['cooldown_seconds']}s")
        
        # Risk Management
        print(f"\n🛡️  RISK MANAGEMENT")
        print("-" * 60)
        print(f"💵 Position Size: ${risk_config['fixed_position_usdt']:,} USDT")
        print(f"🎯 Profit Target: ${risk_config['fixed_break_even_threshold']} USDT")
        print(f"⚡ Leverage: {risk_config['leverage']}x")
        
        # Performance Notes
        
        print("\n" + "=" * 60)
        print("🟢 Bot started successfully - Monitoring for signals...")
        
        # Send Telegram notification with actual config
        config_summary = f"${risk_config['fixed_position_usdt']} USDT @ {risk_config['leverage']}x leverage"
        await self.engine.notifier.send_bot_status("started", f"HF Scalping Mode Active - {config_summary}")
    
    async def _shutdown(self):
        """Shutdown bot gracefully"""
        print("\n🛑 Shutting down...")
        self.running = False
        
        # Close any open positions
        if self.engine.position:
            print("⚠️  Closing open position...")
            await self.engine._close_position("Bot shutdown")
        
        # Show final statistics
        self._show_session_stats()
        
        # Send shutdown notification
        await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
        print("✅ Bot stopped successfully")
    
    def _show_session_stats(self):
        """Show session statistics"""
        try:
            exit_reasons = self.engine.exit_reasons
            rejections = self.engine.rejections
            
            total_trades = sum(exit_reasons.values())
            total_signals = rejections.get('total_signals', 0)
            
            if total_trades > 0 or total_signals > 0:
                print(f"\n📊 SESSION STATISTICS")
                print("-" * 40)
                print(f"🔢 Total Trades: {total_trades}")
                if total_signals > 0:
                    acceptance_rate = (total_trades / total_signals) * 100
                    print(f"📈 Signal Acceptance: {acceptance_rate:.1f}% ({total_trades}/{total_signals})")
                
                if total_trades > 0:
                    print(f"🎯 Top Exit Reasons:")
                    sorted_exits = sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True)
                    for reason, count in sorted_exits[:3]:
                        if count > 0:
                            print(f"   • {reason.replace('_', ' ').title()}: {count}")
        except:
            pass

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    raise KeyboardInterrupt

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    print("⚡ Initializing High-Frequency Scalping Bot...")
    
    try:
        bot = HFScalpingBot()
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print(f"💡 Check your configuration and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()