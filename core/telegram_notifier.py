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
        
        status = "✅ enabled" if self.enabled else "⚠️ disabled (missing credentials)"
        print(f"Telegram notifications {status}")
    
    async def send_message(self, message):
        """Send raw message to Telegram"""
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
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            return False
    
    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        """Send trade entry notification"""
        try:
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
📏 <b>Structure Level:</b> ${signal_data['level']:.2f}
🎯 <b>Confidence:</b> {signal_data.get('confidence', 0):.0f}%

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Trade entry notification error: {e}")
    
    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        """Send trade exit notification"""
        try:
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
        except Exception as e:
            print(f"❌ Trade exit notification error: {e}")
    
    async def send_bot_status(self, status, message_text=""):
        """Send bot status notification"""
        try:
            status_emoji = {
                'started': '🚀', 'stopped': '🛑', 'error': '❌', 'warning': '⚠️'
            }
            
            emoji = status_emoji.get(status, '📊')
            
            message = f"""
{emoji} <b>BOT STATUS: {status.upper()}</b>

📊 <b>Symbol:</b> {self.symbol}
📋 <b>Strategy:</b> RSI/MFI Strategy
{f"💬 <b>Message:</b> {message_text}" if message_text else ""}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Bot status notification error: {e}")
    
    async def send_error_alert(self, error_type, error_message):
        """Send error alert notification"""
        try:
            message = f"""
❌ <b>ERROR ALERT</b>

📊 <b>Symbol:</b> {self.symbol}
🚨 <b>Type:</b> {error_type}
💬 <b>Message:</b> {error_message}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Error alert notification error: {e}")