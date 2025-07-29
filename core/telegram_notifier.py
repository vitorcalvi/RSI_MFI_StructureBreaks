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
            print("ℹ️ Telegram notifications disabled")
            return

        try:
            from telegram import Bot
            self.bot = Bot(token=self.bot_token)
            print("✅ Telegram notifications enabled")
        except ImportError:
            print("⚠️ python-telegram-bot not installed")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ Telegram init failed: {e}")
            self.enabled = False

    async def send_message(self, message):
        if not self.enabled or not self.bot:
            print(f"📱 {message}")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"❌ Telegram error: {e}\n📱 {message}")

    async def trade_opened(self, symbol, price, size, side):
        self.position_start_time = datetime.now()
        direction = "📈 LONG" if side == "Buy" else "📉 SHORT"
        value = size * price
        
        msg = (f"🔔 {direction} {symbol}\n"
               f"💵 ${value:.2f} @ ${price:.4f}\n"
               f"⏰ {self.position_start_time:%H:%M:%S}")
        await self.send_message(msg)

    async def trade_closed(self, symbol, pnl_pct, pnl_usd, reason="Signal"):
        close_time = datetime.now()
        duration = "N/A"
        
        if self.position_start_time:
            minutes = (close_time - self.position_start_time).total_seconds() / 60
            if minutes < 60:
                duration = f"{int(minutes)}m"
            else:
                hours, mins = int(minutes // 60), int(minutes % 60)
                duration = f"{hours}h {mins}m"
            self.position_start_time = None

        status = "✅ PROFIT" if pnl_pct > 0 else "❌ LOSS"
        
        msg = (f"{status} {symbol}\n"
               f"📈 {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
               f"🎯 {reason}\n"
               f"⏱️ {duration} | ⏰ {close_time:%H:%M:%S}")
        await self.send_message(msg)

    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        msg = (f"🔒 PROFIT LOCK!\n"
               f"📊 {symbol} +{pnl_pct:.2f}%\n"
               f"🎯 Trailing: {trailing_pct:.1f}%\n"
               f"⏰ {datetime.now():%H:%M:%S}")
        await self.send_message(msg)

    async def error_notification(self, error_msg):
        msg = f"⚠️ ERROR: {error_msg}\n⏰ {datetime.now():%H:%M:%S}"
        await self.send_message(msg)

    async def bot_started(self, symbol, balance):
        msg = (f"🤖 BOT STARTED\n"
               f"📊 {symbol}\n"
               f"💰 ${balance:.2f}\n"
               f"⏰ {datetime.now():%H:%M:%S}")
        await self.send_message(msg)

    async def bot_stopped(self):
        msg = f"⏹️ BOT STOPPED\n⏰ {datetime.now():%H:%M:%S}"
        await self.send_message(msg)