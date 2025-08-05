import asyncio
import signal
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class HFScalpingBot:
    """High-Frequency Scalping Bot with streamlined startup/shutdown."""
    
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        
    async def start(self):
        """Start the trading bot with simplified flow."""
        if not self._validate_environment():
            return
            
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup()
        await self._run_trading_loop()
        await self._shutdown()
    
    def _validate_environment(self):
        """Validate required environment variables."""
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
        """Display startup information and send notification."""
        balance = await self.engine.get_account_balance()
        strategy_info = self.engine.strategy.get_strategy_info()
        risk_config = self.engine.risk_manager.config
        
        self._print_startup_info(balance, strategy_info, risk_config)
        
        # Send Telegram notification
        config_summary = f"${risk_config['fixed_position_usdt']} USDT @ {risk_config['leverage']}x leverage"
        await self.engine.notifier.send_bot_status("started", f"HF Scalping Mode Active - {config_summary}")
    
    def _print_startup_info(self, balance, strategy_info, risk_config):
        """Print comprehensive startup information."""
        strategy_config = self.engine.strategy.config
        demo_mode = "TESTNET" if self.engine.demo_mode else "LIVE"
        
        print(f"\n🚀 {self.engine.symbol} HIGH-FREQUENCY SCALPING BOT")
        print("=" * 60)
        
        # Environment and account
        print(f"🌐 Environment: {demo_mode}")
        print(f"💰 Account Balance: ${balance:,.2f} USDT")
        print(f"📊 Symbol: {self.engine.symbol}")
        
        # Strategy configuration  
        print(f"\n⚙️  STRATEGY: {strategy_info['name']}")
        print("-" * 60)
        print(f"📈 RSI Length: {strategy_config['rsi_length']} | MFI Length: {strategy_config['mfi_length']}")
        print(f"🎯 Uptrend: RSI≤{strategy_config['uptrend_oversold']}, MFI≤{strategy_config['uptrend_mfi_threshold']}")
        print(f"📉 Neutral Long: RSI≤{strategy_config['neutral_oversold']}, MFI≤{strategy_config['neutral_mfi_threshold']}")
        print(f"⚡ Cooldown: {strategy_config['cooldown_seconds']}s")
        
        # Risk management
        print(f"\n🛡️  RISK MANAGEMENT")
        print("-" * 60)
        print(f"💵 Position Size: ${risk_config['fixed_position_usdt']:,} USDT")
        print(f"🎯 Profit Target: ${risk_config['fixed_break_even_threshold']} USDT")
        print(f"⚡ Leverage: {risk_config['leverage']}x")
        print(f"⏰ Max Hold Time: {risk_config['max_position_time']}s")
        print(f"🚨 Emergency Stop: {risk_config['emergency_stop_pct']*100:.1f}%")
        
        # Performance notes
        print(f"\n📊 PERFORMANCE NOTES")
        print("-" * 60)
        for key, note in strategy_info['performance_notes'].items():
            print(f"• {key.replace('_', ' ').title()}: {note}")
        
        print("\n" + "=" * 60)
        print("🟢 Bot started successfully - Monitoring for signals...")
    
    async def _run_trading_loop(self):
        """Run the main trading loop."""
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
    
    async def _shutdown(self):
        """Shutdown bot gracefully."""
        print("\n🛑 Shutting down...")
        self.running = False
        
        # Close any open positions
        if self.engine.position:
            print("⚠️  Closing open position...")
            await self.engine._close_position("Bot shutdown")
        
        # Show final statistics and send notification
        self._show_session_stats()
        await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
        print("✅ Bot stopped successfully")
    
    def _show_session_stats(self):
        """Show session statistics."""
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
        except Exception:
            pass

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point with error handling."""
    setup_signal_handlers()
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