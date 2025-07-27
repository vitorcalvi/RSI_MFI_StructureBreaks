import asyncio
import os
from telegram import Bot
from datetime import datetime

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            self.bot = Bot(token=self.bot_token)
    
    async def send(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except:
            pass
    
    async def trade_opened(self, symbol, price, size):
        """Notify when trade is opened"""
        message = f"✅ LONG {symbol}\nPrice: ${price:.4f}\nSize: {size:.4f}"
        await self.send(message)
    
    async def trade_closed(self, symbol, pnl_pct, pnl_usd):
        """Notify when trade is closed"""
        emoji = "💚" if pnl_pct > 0 else "💔"
        message = f"{emoji} CLOSED {symbol}\nP&L: {pnl_pct:+.2f}% (${pnl_usd:.2f})"
        await self.send(message)