import asyncio
import os
from telegram import Bot

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
        except Exception as e:
            print(f"Telegram error: {e}")
    
    
    async def trade_opened(self, symbol, price, size, is_long=True):
        """Notify when trade is opened"""
        side = "LONG" if is_long else "SHORT"
        message = (
            f"🔔 {side} {symbol}\n"
            f"💰 ${price:,.2f}\n"
            f"📊 {size:.4f}"
        )
        await self.send(message)

    async def trade_closed(self, symbol, pnl_pct, pnl_usd, is_long=True):
        """Notify when trade is closed"""
        side = "LONG" if is_long else "SHORT"
        profit = pnl_pct > 0
        message = (
            f"{'✅' if profit else '❌'} CLOSED {side} {symbol}\n"
            f"📈 {pnl_pct:+.2f}%\n"
            f"💵 ${pnl_usd:+,.2f}"
        )
        await self.send(message)