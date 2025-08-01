import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        print("✅ Telegram notifications enabled" if self.enabled else "⚠️  Telegram notifications disabled (missing credentials)")
    
    async def send_message(self, message):
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
    
    async def send_trade_signal(self, signal, price, quantity):
        try:
            emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
            direction = "LONG" if signal['action'] == 'BUY' else "SHORT"
            
            message = f"""
{emoji} <b>TRADE SIGNAL - {direction}</b>

📊 <b>Symbol:</b> ETHUSDT
💰 <b>Price:</b> ${price:.2f}
📈 <b>Quantity:</b> {quantity}
🛑 <b>Stop Loss:</b> ${signal['structure_stop']:.2f}

📋 <b>Strategy:</b> {signal['signal_type']}
"""
            
            if 'rsi' in signal and 'mfi' in signal:
                message += f"📊 <b>RSI:</b> {signal['rsi']} | <b>MFI:</b> {signal['mfi']}\n"
            
            if 'level' in signal:
                message += f"📏 <b>Structure Level:</b> ${signal['level']:.2f}\n"
            
            message += f"\n⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Trade signal notification error: {e}")
    
    async def send_position_update(self, position_data):
        try:
            side = position_data.get('side', 'Unknown')
            size = position_data.get('size', '0')
            entry_price = float(position_data.get('avgPrice', 0))
            unrealized_pnl = float(position_data.get('unrealisedPnl', 0))
            
            position_value = float(size) * entry_price
            pnl_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
            emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            
            message = f"""
{emoji} <b>POSITION UPDATE</b>

📊 <b>Symbol:</b> ETHUSDT
📈 <b>Side:</b> {side}
💰 <b>Size:</b> {size}
💵 <b>Entry:</b> ${entry_price:.2f}
📊 <b>PnL:</b> ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Position update notification error: {e}")
    
    async def send_position_closed(self, final_pnl=None):
        try:
            message = f"""
🏁 <b>POSITION CLOSED</b>

📊 <b>Symbol:</b> ETHUSDT
"""
            
            if final_pnl is not None:
                emoji = "✅" if final_pnl >= 0 else "❌"
                message += f"{emoji} <b>Final PnL:</b> ${final_pnl:.2f}\n"
            
            message += f"\n⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Position closed notification error: {e}")
    
    async def send_profit_lock(self, new_stop, current_pnl):
        try:
            message = f"""
🔒 <b>PROFIT LOCK ACTIVATED</b>

📊 <b>Symbol:</b> ETHUSDT
🛑 <b>New Stop:</b> ${new_stop:.2f}
💰 <b>Current PnL:</b> ${current_pnl:.2f}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Profit lock notification error: {e}")
    
    async def send_error_alert(self, error_type, error_message):
        try:
            message = f"""
⚠️ <b>BOT ERROR ALERT</b>

🔧 <b>Type:</b> {error_type}
📝 <b>Message:</b> {error_message}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Error alert notification error: {e}")
    
    async def send_bot_status(self, status, additional_info=None):
        try:
            status_emoji = {
                'started': '🚀',
                'stopped': '🛑',
                'connected': '✅',
                'disconnected': '❌',
                'error': '⚠️'
            }
            
            emoji = status_emoji.get(status.lower(), '📊')
            
            message = f"""
{emoji} <b>BOT STATUS: {status.upper()}</b>

📊 <b>Symbol:</b> ETHUSDT
🔄 <b>Strategy:</b> RSI + MFI + Break & Retest
"""
            
            if additional_info:
                message += f"📝 <b>Info:</b> {additional_info}\n"
            
            message += f"\n⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Bot status notification error: {e}")
    
    async def send_daily_summary(self, trades_count, total_pnl, win_rate):
        try:
            emoji = "📈" if total_pnl >= 0 else "📉"
            
            message = f"""
{emoji} <b>DAILY TRADING SUMMARY</b>

📊 <b>Symbol:</b> ETHUSDT
🔢 <b>Total Trades:</b> {trades_count}
💰 <b>Total PnL:</b> ${total_pnl:.2f}
🎯 <b>Win Rate:</b> {win_rate:.1f}%

📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
"""
            
            await self.send_message(message)
        except Exception as e:
            print(f"❌ Daily summary notification error: {e}")
    
    def test_connection(self):
        if not self.enabled:
            print("❌ Telegram not configured")
            return False
        
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': '✅ Telegram bot connection test successful!',
                    'parse_mode': 'HTML'
                },
                timeout=10
            )
            
            success = response.status_code == 200
            print("✅ Telegram connection test successful" if success else f"❌ Telegram test failed: {response.status_code}")
            return success
        except Exception as e:
            print(f"❌ Telegram test error: {e}")
            return False