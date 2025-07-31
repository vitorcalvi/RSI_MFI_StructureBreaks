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
        try:
            self.position_start_time = datetime.now()
            direction = "📈 LONG" if side == "Buy" else "📉 SHORT"
            
            # Safe value handling
            price = float(price) if price is not None else 0.0
            size = float(size) if size is not None else 0.0
            value = size * price
            
            symbol_short = symbol.replace('/', '') if symbol else 'ETHUSDT'
            
            msg = (f"{direction} {symbol_short}\n"
                   f"💵 ${value:.0f} @ ${price:.2f}\n"
                   f"💸 Risk: $100 fixed\n"
                   f"⏰ {self.position_start_time:%H:%M:%S}")
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Trade opened notification error: {e}")

    async def trade_closed(self, symbol, pnl_pct, pnl_usd, reason="Signal"):
        try:
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

            # Safe value handling
            pnl_usd = float(pnl_usd) if pnl_usd is not None else 0.0
            status = "✅ WIN" if pnl_usd > 0 else "❌ LOSS"
            symbol_short = symbol.replace('/', '') if symbol else 'ETHUSDT'
            
            msg = (f"{status} {symbol_short}\n"
                   f"💰 ${pnl_usd:+.2f}\n"
                   f"🎯 {reason}\n"
                   f"⏱️ {duration} | ⏰ {close_time:%H:%M:%S}")
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Trade closed notification error: {e}")

    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        try:
            # Safe value handling
            pnl_pct = float(pnl_pct) if pnl_pct is not None else 0.0
            trailing_pct = float(trailing_pct) if trailing_pct is not None else 1.0
            
            symbol_short = symbol.replace('/', '') if symbol else 'ETHUSDT'
            
            msg = (f"🔒 PROFIT LOCK!\n"
                   f"📊 {symbol_short}\n"
                   f"🎯 Trailing: {trailing_pct:.1f}%\n"
                   f"⏰ {datetime.now():%H:%M:%S}")
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Profit lock notification error: {e}")

    async def error_notification(self, error_msg):
        try:
            # Safe error message handling
            error_msg = str(error_msg) if error_msg is not None else "Unknown error"
            
            msg = f"⚠️ ERROR: {error_msg}\n⏰ {datetime.now():%H:%M:%S}"
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Error notification failed: {e}")

    async def bot_started(self, symbol, balance):
        try:
            # Safe value handling - THIS WAS CAUSING THE ERROR
            balance = float(balance) if balance is not None else 0.0
            symbol_short = symbol.replace('/', '') if symbol else 'ETHUSDT'
            
            msg = (f"🤖 BOT STARTED\n"
                   f"📊 {symbol_short}\n"
                   f"💰 ${balance:.0f}\n"  # Now safe - balance is guaranteed to be a float
                   f"💸 Risk: $100/trade\n"
                   f"⏰ {datetime.now():%H:%M:%S}")
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Bot started notification error: {e}")

    async def bot_stopped(self):
        try:
            msg = f"⏹️ BOT STOPPED\n⏰ {datetime.now():%H:%M:%S}"
            await self.send_message(msg)
        except Exception as e:
            print(f"⚠️ Bot stopped notification error: {e}")