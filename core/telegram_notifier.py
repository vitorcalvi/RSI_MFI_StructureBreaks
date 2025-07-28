import asyncio
import os
from datetime import datetime

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.position_start_time = None
        self.bot = None

        if not self.enabled:
            print("ℹ️ Telegram notifications disabled (no credentials)")
            return

        try:
            from telegram import Bot
            self.bot = Bot(token=self.bot_token)
            print("✅ Telegram notifications enabled")
        except ImportError:
            print("⚠️ python-telegram-bot not installed - notifications disabled")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ Telegram initialization failed: {e}")
            self.enabled = False

    async def send_message(self, message):
        """Send message to Telegram or print to console."""
        if not self.enabled or not self.bot:
            print(f"📱 {message}")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"❌ Telegram error: {e}\n📱 Message: {message}")

    async def trade_opened(self, symbol, price, size, side, potential_gain=None, potential_loss=None):
        """FIXED: Set position start time for duration calculation"""
        self.position_start_time = datetime.now()  # CRITICAL FIX: Set start time
        
        direction_emoji = "📈" if side == "Buy" else "📉"
        position_value = size * price
        msg = (
            f"🔔 {direction_emoji} OPENED {symbol}\n"
            f"📍 Direction: {side.upper()}\n"
            f"⏰ {self.position_start_time:%H:%M:%S}\n"  # Use stored time
            f"💰 Price: ${price:.4f}\n"
            f"📊 Size: {size}\n"
            f"💵 Value: ${position_value:.2f} USDT"
        )
        if potential_gain:
            msg += f"\n🎯 Target: +${potential_gain:.2f}"
        if potential_loss:
            msg += f"\n🛡️ Max Loss: -${potential_loss:.2f}"
        await self.send_message(msg)

    async def trade_closed(self, symbol, pnl_pct, pnl_usd, reason="Signal"):
        """ENHANCED: Better duration calculation and profit/loss tracking"""
        close_time = datetime.now()
        duration_str = "N/A"
        earn_per_hour = 0

        if self.position_start_time:
            minutes = (close_time - self.position_start_time).total_seconds() / 60
            if minutes < 60:
                duration_str = f"{int(minutes)}m"
            else:
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                duration_str = f"{hours}h {mins}m"
            
            # Calculate hourly rate
            earn_per_hour = (pnl_usd * 60) / minutes if minutes > 0 else 0
            
            # Reset position start time
            self.position_start_time = None

        is_profit = pnl_pct > 0
        status_emoji = "✅ 💰" if is_profit else "❌ 📉"
        profit_status = "PROFIT" if is_profit else "LOSS"

        reason_icons = {
            "Signal": "🎯", 
            "Reverse Signal": "🔄", 
            "Loss Limit": "🚨",
            "Bot Stop": "⏹️", 
            "Take Profit": "💰", 
            "Stop Loss": "🛡️", 
            "Trailing Stop": "🔒",
            "Profit Protection": "🛡️💰"  # Added for profit protection
        }
        icon = reason_icons.get(reason, "📝")

        msg = (
            f"{status_emoji} CLOSED {symbol} - {profit_status}\n"
            f"{icon} Reason: {reason}\n"
            f"⏰ Closed: {close_time:%H:%M:%S}\n"
            f"⏱️ Duration: {duration_str}\n"
            f"📈 P&L: {pnl_pct:+.2f}%\n"
            f"💵 Amount: ${pnl_usd:+.2f} USDT\n"
            f"📊 Rate: ${earn_per_hour:+.2f}/hour"
        )
        await self.send_message(msg)

    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        """Notify when profit lock is activated"""
        msg = (
            f"🔒 💎 PROFIT LOCK ACTIVATED!\n"
            f"📊 Symbol: {symbol}\n"
            f"📈 Current P&L: +{pnl_pct:.2f}%\n"
            f"🎯 Trailing Stop: {trailing_pct:.1f}%\n"
            f"💰 Protecting Profits Now!\n"
            f"⏰ {datetime.now():%H:%M:%S}"
        )
        await self.send_message(msg)

    async def position_switched(self, symbol, from_side, to_side, size, pnl_pct, pnl_usd):
        """Notify when position is switched due to losses"""
        msg = (
            f"🔄 ⚡ POSITION SWITCHED!\n"
            f"📊 Symbol: {symbol}\n"
            f"🔀 From: {from_side} → {to_side}\n"
            f"📈 Size: {size}\n"
            f"📉 Loss Cut: {pnl_pct:.2f}% (${pnl_usd:.2f})\n"
            f"🎯 New Direction: {to_side.upper()}\n"
            f"⏰ {datetime.now():%H:%M:%S}"
        )
        await self.send_message(msg)

    async def trailing_stop_updated(self, symbol, new_stop, current_price):
        """Notify when trailing stop is updated"""
        msg = (
            f"🔄 🔒 TRAILING UPDATED\n"
            f"📊 {symbol}\n"
            f"🎯 New Stop: ${new_stop:.4f}\n"
            f"💰 Current: ${current_price:.4f}\n"
            f"⏰ {datetime.now():%H:%M:%S}"
        )
        await self.send_message(msg)

    async def error_notification(self, error_msg):
        """Notify about system errors"""
        msg = f"⚠️ SYSTEM ERROR\n❌ {error_msg}\n⏰ {datetime.now():%H:%M:%S}"
        await self.send_message(msg)

    async def bot_started(self, symbol, balance):
        """Notify when bot starts"""
        msg = (
            f"🤖 TRADING BOT STARTED\n"
            f"📊 Symbol: {symbol}\n"
            f"💰 Balance: ${balance:.2f} USDT\n"
            f"⏰ {datetime.now():%H:%M:%S}"
        )
        await self.send_message(msg)

    async def bot_stopped(self):
        """Notify when bot stops"""
        msg = f"⏹️ TRADING BOT STOPPED\n⏰ {datetime.now():%H:%M:%S}"
        await self.send_message(msg)