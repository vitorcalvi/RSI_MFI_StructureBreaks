import asyncio
import os
from datetime import datetime
from telegram import Bot

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.position_start_time = None
        if self.enabled:
            self.bot = Bot(token=self.bot_token)
    
    async def send_message(self, message):
        if not self.enabled:
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")
    
    async def trade_opened(self, symbol, price, size, potential_gain=None, potential_loss=None):
        self.position_start_time = datetime.now()
        message = f"🔔 OPENED {symbol}\n⏰ {self.position_start_time.strftime('%H:%M:%S')}\nPrice: ${price:.4f}\nSize: {size}\nValue: ${size * price:.2f}"
        if potential_gain is not None:
            message += f"\nPotential Gains: {potential_gain} USD"
        if potential_loss is not None:
            message += f"\nPotential Losses: {potential_loss} USD"
        await self.send_message(message)
    
    async def trade_closed(self, symbol, pnl_pct, pnl_usd):
        close_time = datetime.now()
        duration_str = "N/A"
        earn_per_hour = 0
        
        if self.position_start_time:
            total_minutes = (close_time - self.position_start_time).total_seconds() / 60
            duration_str = f"{int(total_minutes)}m" if total_minutes < 60 else f"{int(total_minutes // 60)}h {int(total_minutes % 60)}m"
            if total_minutes > 0:
                earn_per_hour = (pnl_usd * 60) / total_minutes
        
        message = f"{'✅' if pnl_pct > 0 else '❌'} CLOSED {symbol}\n⏰ {close_time.strftime('%H:%M:%S')}\n⏱️ Duration: {duration_str}\n📈 {pnl_pct:+.2f}%\n💵 ${pnl_usd:+.2f}\n📊 ${earn_per_hour:+.2f}/hour"
        await self.send_message(message)
    
    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        message = f"🔒 PROFIT LOCK ACTIVATED!\nSymbol: {symbol}\nP&L: {pnl_pct:.2f}%\nTrailing Stop: {trailing_pct}%\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        await self.send_message(message)
    
    async def trailing_stop_updated(self, symbol, new_stop, current_price):
        message = f"🔄 TRAILING STOP UPDATED\nSymbol: {symbol}\nNew Stop: ${new_stop:.4f}\nCurrent: ${current_price:.4f}\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        await self.send_message(message)