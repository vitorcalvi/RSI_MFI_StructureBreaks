import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
        self.enabled = bool(self.bot_token and self.chat_id)
    
    async def send_message(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        """Send trade entry notification"""
        emoji = "🟢" if signal_data['action'] == 'BUY' else "🔴"
        direction = "LONG" if signal_data['action'] == 'BUY' else "SHORT"
        
        message = f"""
{emoji} <b>TRADE ENTRY - {direction}</b>

📊 <b>Symbol:</b> {self.symbol}
💰 <b>Price:</b> ${price:.2f}
📈 <b>Quantity:</b> {quantity}
🛑 <b>Stop Loss:</b> ${signal_data['structure_stop']:.2f}

📋 <b>Strategy:</b> {signal_data['signal_type']}
📊 <b>RSI:</b> {signal_data['rsi']:.1f} | <b>MFI:</b> {signal_data['mfi']:.1f}
🎯 <b>Confidence:</b> {signal_data.get('confidence', 0):.0f}%

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)
    
    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        """Send trade exit notification"""
        emoji = "🟢" if pnl >= 0 else "🔴"
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        trigger_formatted = exit_data['trigger'].replace('_', ' ').title()
        
        message = f"""
{emoji} <b>TRADE EXIT</b>

📊 <b>Symbol:</b> {self.symbol}
💰 <b>Exit Price:</b> ${price:.2f}
💵 <b>PnL:</b> {pnl_text}
⏱️ <b>Duration:</b> {duration:.1f}s

🔄 <b>Trigger:</b> {trigger_formatted}
📊 <b>RSI:</b> {exit_data.get('rsi', 0):.1f} | <b>MFI:</b> {exit_data.get('mfi', 0):.1f}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)
    
    async def send_bot_status(self, status, message_text=""):
        """Send bot status notification"""
        status_emojis = {
            'started': '🚀', 'stopped': '🛑', 
            'error': '❌', 'warning': '⚠️'
        }
        
        emoji = status_emojis.get(status, '📊')
        
        message = f"""
{emoji} <b>BOT STATUS: {status.upper()}</b>

📊 <b>Symbol:</b> {self.symbol}
📋 <b>Strategy:</b> RSI/MFI Strategy
{f"💬 <b>Message:</b> {message_text}" if message_text else ""}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)