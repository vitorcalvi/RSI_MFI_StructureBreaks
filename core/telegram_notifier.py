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
    
    async def send(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")
    
    async def trade_opened(self, symbol, price, size):
        """Notify when trade is opened"""
        self.position_start_time = datetime.now()
        position_value = size * price
        
        message = (
            f"🔔 OPENED {symbol}\n"
            f"⏰ {self.position_start_time.strftime('%H:%M:%S')}\n"
            f"Price: ${price:.4f}\n"
            f"Size: {size}\n"
            f"Value: ${position_value:.2f}"
        )
        
        await self.send(message)
    
    async def trade_closed(self, symbol, pnl_pct, pnl_usd):
        """Notify when trade is closed"""
        close_time = datetime.now()
        profit = pnl_pct > 0
        
        # Calculate duration
        duration_str = "N/A"
        earn_per_hour = 0
        
        if self.position_start_time:
            duration = close_time - self.position_start_time
            total_minutes = duration.total_seconds() / 60
            
            if total_minutes < 60:
                duration_str = f"{int(total_minutes)}m"
            else:
                hours = int(total_minutes // 60)
                minutes = int(total_minutes % 60)
                duration_str = f"{hours}h {minutes}m"
            
            # Calculate earn/loss per hour
            if total_minutes > 0:
                earn_per_hour = (pnl_usd * 60) / total_minutes
        
        message = (
            f"{'✅' if profit else '❌'} CLOSED {symbol}\n"
            f"⏰ {close_time.strftime('%H:%M:%S')}\n"
            f"⏱️ Duration: {duration_str}\n"
            f"📈 {pnl_pct:+.2f}%\n"
            f"💵 ${pnl_usd:+.2f}\n"
            f"📊 ${earn_per_hour:+.2f}/hour"
        )
        await self.send(message)