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
        
        if self.enabled:
            try:
                # Try modern python-telegram-bot first
                from telegram import Bot
                self.bot = Bot(token=self.bot_token)
                print("✅ Telegram notifications enabled")
            except ImportError:
                try:
                    # Fallback for older versions
                    import telegram
                    self.bot = telegram.Bot(token=self.bot_token)
                    print("✅ Telegram notifications enabled (legacy)")
                except ImportError:
                    print("⚠️ python-telegram-bot not installed - notifications disabled")
                    self.enabled = False
            except Exception as e:
                print(f"⚠️ Telegram initialization failed: {e}")
                self.enabled = False
        else:
            print("ℹ️ Telegram notifications disabled (no credentials)")
    
    async def send_message(self, message):
        """Send message to Telegram or print to console"""
        if not self.enabled or not self.bot:
            print(f"📱 {message}")
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            print(f"📱 Message: {message}")
    
    async def trade_opened(self, symbol, price, size, side, potential_gain=None, potential_loss=None):
        """Notify when trade is opened"""
        self.position_start_time = datetime.now()
        position_value = size * price
        
        direction_emoji = "📈" if side == "Buy" else "📉"
        
        message = (
            f"🔔 {direction_emoji} OPENED {symbol}\n"
            f"📍 Direction: {side.upper()}\n"
            f"⏰ {self.position_start_time.strftime('%H:%M:%S')}\n"
            f"💰 Price: ${price:.4f}\n"
            f"📊 Size: {size}\n"
            f"💵 Value: ${position_value:.2f} USDT"
        )
        
        if potential_gain:
            message += f"\n🎯 Target: +${potential_gain:.2f}"
        if potential_loss:
            message += f"\n🛡️ Max Loss: -${potential_loss:.2f}"
            
        await self.send_message(message)
    
    async def trade_closed(self, symbol, pnl_pct, pnl_usd, reason="Signal"):
        """Notify when trade is closed"""
        close_time = datetime.now()
        duration_str = "N/A"
        earn_per_hour = 0
        
        if self.position_start_time:
            total_minutes = (close_time - self.position_start_time).total_seconds() / 60
            if total_minutes < 60:
                duration_str = f"{int(total_minutes)}m"
            else:
                hours = int(total_minutes // 60)
                mins = int(total_minutes % 60)
                duration_str = f"{hours}h {mins}m"
            
            if total_minutes > 0:
                earn_per_hour = (pnl_usd * 60) / total_minutes
        
        # Status
        is_profit = pnl_pct > 0
        status_emoji = "✅ 💰" if is_profit else "❌ 📉"
        profit_status = "PROFIT" if is_profit else "LOSS"
        
        # Reason icons
        reason_icons = {
            "Signal": "🎯",
            "Reverse Signal": "🔄",
            "Loss Limit": "🚨",
            "Bot Stop": "⏹️",
            "Take Profit": "💰",
            "Stop Loss": "🛡️",
            "Trailing Stop": "🔒"
        }
        
        icon = reason_icons.get(reason, "📝")
        
        message = (
            f"{status_emoji} CLOSED {symbol} - {profit_status}\n"
            f"{icon} Reason: {reason}\n"
            f"⏰ Closed: {close_time.strftime('%H:%M:%S')}\n"
            f"⏱️ Duration: {duration_str}\n"
            f"📈 P&L: {pnl_pct:+.2f}%\n"
            f"💵 Amount: ${pnl_usd:+.2f} USDT\n"
            f"📊 Rate: ${earn_per_hour:+.2f}/hour"
        )
        await self.send_message(message)
    
    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        """Notify when profit lock is activated"""
        message = (
            f"🔒 💎 PROFIT LOCK ACTIVATED!\n"
            f"📊 Symbol: {symbol}\n"
            f"📈 Current P&L: +{pnl_pct:.2f}%\n"
            f"🎯 Trailing Stop: {trailing_pct:.1f}%\n"
            f"💰 Protecting Profits Now!\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def position_switched(self, symbol, from_side, to_side, size, pnl_pct, pnl_usd):
        """Notify when position is switched due to losses"""
        switch_time = datetime.now()
        
        message = (
            f"🔄 ⚡ POSITION SWITCHED!\n"
            f"📊 Symbol: {symbol}\n"
            f"🔀 From: {from_side} → {to_side}\n"
            f"📈 Size: {size}\n"
            f"📉 Loss Cut: {pnl_pct:.2f}% (${pnl_usd:.2f})\n"
            f"🎯 New Direction: {to_side.upper()}\n"
            f"⏰ {switch_time.strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def trailing_stop_updated(self, symbol, new_stop, current_price):
        """Notify when trailing stop is updated"""
        message = (
            f"🔄 🔒 TRAILING UPDATED\n"
            f"📊 {symbol}\n"
            f"🎯 New Stop: ${new_stop:.4f}\n"
            f"💰 Current: ${current_price:.4f}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def error_notification(self, error_msg):
        """Notify about system errors"""
        message = (
            f"⚠️ SYSTEM ERROR\n"
            f"❌ {error_msg}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def bot_started(self, symbol, balance):
        """Notify when bot starts"""
        message = (
            f"🤖 TRADING BOT STARTED\n"
            f"📊 Symbol: {symbol}\n"
            f"💰 Balance: ${balance:.2f} USDT\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def bot_stopped(self):
        """Notify when bot stops"""
        message = (
            f"⏹️ TRADING BOT STOPPED\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)