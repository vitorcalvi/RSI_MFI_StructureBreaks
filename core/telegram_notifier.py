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
    
    async def trade_opened(self, symbol, price, size, side, potential_gain=None, potential_loss=None):
        self.position_start_time = datetime.now()
        position_value = size * price
        
        # Position direction emoji
        direction_emoji = "📈" if side == "Buy" else "📉"
        
        message = (
            f"🔔 {direction_emoji} OPENED {symbol}\n"
            f"📍 Direction: {side.upper()}\n"
            f"⏰ {self.position_start_time.strftime('%H:%M:%S')}\n"
            f"💰 Price: ${price:.4f}\n"
            f"📊 Size: {size}\n"
            f"💵 Value: ${position_value:.2f} USDT"
        )
        
        if potential_gain is not None:
            message += f"\n🎯 Target Profit: ${potential_gain:.2f}"
        if potential_loss is not None:
            message += f"\n🛡️ Max Loss: ${potential_loss:.2f}"
            
        await self.send_message(message)
    
    async def trade_closed(self, symbol, pnl_pct, pnl_usd, reason="Signal"):
        close_time = datetime.now()
        duration_str = "N/A"
        earn_per_hour = 0
        
        if self.position_start_time:
            total_minutes = (close_time - self.position_start_time).total_seconds() / 60
            duration_str = f"{int(total_minutes)}m" if total_minutes < 60 else f"{int(total_minutes // 60)}h {int(total_minutes % 60)}m"
            if total_minutes > 0:
                earn_per_hour = (pnl_usd * 60) / total_minutes
        
        # Profit/Loss status
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
        message = (
            f"🔒 💎 PROFIT LOCK ACTIVATED!\n"
            f"📊 Symbol: {symbol}\n"
            f"📈 Current P&L: +{pnl_pct:.2f}%\n"
            f"🎯 Trailing Stop: {trailing_pct}%\n"
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
        message = (
            f"🔄 🔒 TRAILING UPDATED\n"
            f"📊 {symbol}\n"
            f"🎯 New Stop: ${new_stop:.4f}\n"
            f"💰 Current: ${current_price:.4f}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)