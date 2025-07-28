import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import bot components
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
from core.risk_management import RiskManager
import asyncio
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
        
        await self.send_message(message)
    
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
        await self.send_message(message)
    
    async def profit_lock_activated(self, symbol, pnl_pct, trailing_pct):
        """Notify when profit lock is activated"""
        message = (
            f"🔒 PROFIT LOCK ACTIVATED!\n"
            f"Symbol: {symbol}\n"
            f"P&L: {pnl_pct:.2f}%\n"
            f"Trailing Stop: {trailing_pct}%\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)
    
    async def trailing_stop_updated(self, symbol, new_stop, current_price):
        """Notify when trailing stop is updated"""
        message = (
            f"🔄 TRAILING STOP UPDATED\n"
            f"Symbol: {symbol}\n"
            f"New Stop: ${new_stop:.4f}\n"
            f"Current: ${current_price:.4f}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)


class TradeEngine:
    def __init__(self):
        # Configuration
        self.symbol = os.getenv('SYMBOLS', 'SOL/USDT')
        self.linear = self.symbol.replace('/', '')
        self.timeframe = '5'  # 5 minutes
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API credentials
        if self.demo_mode:
            self.api_key = os.getenv('TESTNET_BYBIT_API_KEY')
            self.api_secret = os.getenv('TESTNET_BYBIT_API_SECRET')
        else:
            self.api_key = os.getenv('LIVE_BYBIT_API_KEY')
            self.api_secret = os.getenv('LIVE_BYBIT_API_SECRET')
        
        # Components
        self.exchange = None
        self.strategy = RSIMFICloudStrategy()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        
        # State
        self.running = False
        self.position = None
        self.last_signal_time = None
        
        # Profit Locker
        self.break_even_pct = 0.11  # 0.11% break-even threshold
        self.profit_lock_active = False
        self.trailing_stop_distance = 0.5  # 0.5% trailing stop distance
        
    def connect(self):
        """Initialize exchange connection"""
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Test connection
            server_time = self.exchange.get_server_time()
            if server_time.get('retCode') == 0:
                print(f"✅ Connected to Bybit {'Testnet' if self.demo_mode else 'Live'}")
                return True
            else:
                print(f"❌ Connection failed: {server_time.get('retMsg')}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_market_data(self):
        """Fetch latest market data"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.linear,
                interval=self.timeframe,
                limit=100
            )
            
            if klines.get('retCode') != 0:
                print(f"❌ Kline error: {klines.get('retMsg')}")
                return None
            
            data = klines['result']['list']
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return None
    
    def get_account_balance(self):
        """Get USDT balance"""
        try:
            resp = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if resp.get('retCode') == 0:
                for coin in resp['result']['list'][0].get('coin', []):
                    if coin.get('coin') == 'USDT':
                        return float(coin.get('walletBalance', 0))
            return 0
        except Exception as e:
            print(f"❌ Balance error: {e}")
            return 0
    
    def check_position(self):
        """Check current position"""
        try:
            pos_resp = self.exchange.get_positions(category="linear", symbol=self.linear)
            if pos_resp.get('retCode') == 0 and pos_resp['result']['list']:
                position = pos_resp['result']['list'][0]
                if float(position['size']) > 0:
                    self.position = {
                        'side': position['side'],
                        'size': float(position['size']),
                        'avg_price': float(position['avgPrice']),
                        'unrealized_pnl': float(position['unrealisedPnl']),
                        'unrealized_pnl_pct': float(position['unrealisedPnl']) / (float(position['avgPrice']) * float(position['size'])) * 100
                    }
                    return self.position
            self.position = None
            self.profit_lock_active = False  # Reset profit lock when no position
            return None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            return None
    
    def get_symbol_info(self):
        """Get symbol trading rules"""
        try:
            resp = self.exchange.get_instruments_info(category="linear", symbol=self.linear)
            if resp.get('retCode') == 0 and resp['result']['list']:
                info = resp['result']['list'][0]
                return {
                    'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                    'qty_step': float(info['lotSizeFilter']['qtyStep']),
                    'tick_size': float(info['priceFilter']['tickSize'])
                }
            return None
        except Exception as e:
            print(f"❌ Symbol info error: {e}")
            return None
    
    def format_qty(self, info, raw_qty):
        """Format quantity according to exchange requirements"""
        step = info['qty_step']
        qty = float(int(raw_qty / step) * step)
        qty = max(qty, info['min_qty'])
        
        # Determine decimal places
        step_str = f"{step:g}"
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        return f"{qty:.{decimals}f}" if decimals else str(int(qty))
    
    def format_price(self, info, price):
        """Format price according to tick size"""
        tick = info['tick_size']
        price = round(price / tick) * tick
        
        # Determine decimal places
        tick_str = f"{tick:.20f}".rstrip('0').rstrip('.')
        decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{decimals}f}"
    
    def calculate_trailing_stop(self, current_price):
        """Calculate trailing stop loss price"""
        if not self.position:
            return None
            
        side = self.position['side']
        distance_pct = self.trailing_stop_distance / 100
        
        if side == 'Buy':
            # Long position: trailing stop below current price
            trailing_stop = current_price * (1 - distance_pct)
        else:
            # Short position: trailing stop above current price
            trailing_stop = current_price * (1 + distance_pct)
            
        return trailing_stop



    async def update_trailing_stop(self, current_price):
        """Update trailing stop loss - FIXED VERSION"""
        try:
            if not self.position or not self.profit_lock_active:
                return False
                
            side = self.position['side']
            avg_price = self.position['avg_price']
            
            # Skip if current price equals average price (no movement)
            if abs(current_price - avg_price) < 0.01:
                return False
            
            # Calculate trailing stop distance (absolute price, not percentage)
            distance_pct = self.trailing_stop_distance / 100
            
            if side == 'Buy':
                # For Buy positions: trailing stop below current price
                trailing_stop_price = current_price - (current_price * distance_pct)
                
                # Validation: trailing stop must be below current price AND below average price
                if trailing_stop_price >= current_price or trailing_stop_price >= avg_price:
                    return False
                    
                # Active price must be significantly above average price for API validation
                active_price = max(current_price, avg_price * 1.02)  # 2% above average
                
            else:
                # For Sell positions: trailing stop above current price  
                trailing_stop_price = current_price + (current_price * distance_pct)
                
                # Validation: trailing stop must be above current price AND above average price
                if trailing_stop_price <= current_price or trailing_stop_price <= avg_price:
                    return False
                    
                # Active price must be significantly below average price for API validation
                active_price = min(current_price, avg_price * 0.98)  # 2% below average
            
            info = self.get_symbol_info()
            if not info:
                return False
                
            # Format prices with proper precision
            formatted_trailing_stop = self.format_price(info, trailing_stop_price)
            formatted_active_price = self.format_price(info, active_price)
            
            # Additional validation - ensure 10% minimum difference for Bybit API
            price_diff_pct = abs(float(formatted_trailing_stop) - avg_price) / avg_price * 100
            if price_diff_pct < 0.1:  # Less than 0.1% difference
                return False
            
            # Use correct Bybit API parameters for trailing stop
            trailing_params = {
                "category": "linear",
                "symbol": self.linear,
                "positionIdx": 0,
                "trailingStop": formatted_trailing_stop,
                "activePrice": formatted_active_price
            }
            
            # Only update if there's meaningful price movement
            resp = self.exchange.set_trading_stop(**trailing_params)
            
            if resp.get('retCode') == 0:
                return True
            else:
                # Silently fail for validation errors to avoid spam
                return False
                
        except Exception as e:
            # Silent fail to avoid error spam
            return False

    async def check_profit_lock(self, current_price):
        """Check if we should activate profit lock mode - FIXED VERSION"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        
        # Check if we reached break-even
        if not self.profit_lock_active and pnl_pct >= self.break_even_pct:
            self.profit_lock_active = True
            print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}%")
            
            # Send notification
            await self.notifier.profit_lock_activated(
                self.symbol, 
                pnl_pct, 
                self.trailing_stop_distance
            )
            
            # Remove take profit and set trailing stop
            try:
                info = self.get_symbol_info()
                if info:
                    side = self.position['side']
                    distance_pct = self.trailing_stop_distance / 100
                    
                    if side == 'Buy':
                        trailing_stop_price = current_price * (1 - distance_pct)
                        active_price = current_price * 1.01  # 1% above current
                    else:
                        trailing_stop_price = current_price * (1 + distance_pct)
                        active_price = current_price * 0.99  # 1% below current
                    
                    formatted_trailing_stop = self.format_price(info, trailing_stop_price)
                    formatted_active_price = self.format_price(info, active_price)
                    
                    # Remove TP and set trailing stop
                    resp = self.exchange.set_trading_stop(
                        category="linear",
                        symbol=self.linear,
                        positionIdx=0,
                        takeProfit="",  # Remove take profit
                        trailingStop=formatted_trailing_stop,
                        activePrice=formatted_active_price
                    )
                    
                    if resp.get('retCode') == 0:
                        print(f"✅ Trailing stop set: ${formatted_trailing_stop}")
                    else:
                        print(f"❌ Failed to set trailing stop: {resp.get('retMsg')}")
                        
            except Exception as e:
                print(f"❌ Profit lock setup error: {e}")
            
            # Initialize tracking
            self.last_trailing_update = datetime.now()
        
        # Update trailing stop periodically if already active
        elif self.profit_lock_active:
            current_time = datetime.now()
            if not hasattr(self, 'last_trailing_update'):
                self.last_trailing_update = current_time
            
            time_diff = (current_time - self.last_trailing_update).total_seconds()
            
            if time_diff >= 60:  # Update every 60 seconds
                success = await self.update_trailing_stop(current_price)
                if success:
                    print(f"🔒 Trailing updated")
                    self.last_trailing_update = current_time
                    
    async def open_position(self, signal):
        """Open a new position based on signal"""
        try:
            # Get balance and current price
            balance = self.get_account_balance()
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(balance, current_price)
            
            # Get symbol info
            info = self.get_symbol_info()
            if not info:
                return False
            
            # Format quantity
            qty = self.format_qty(info, position_size)
            
            # Determine side
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            print(f"\n📈 Opening {side} position: {qty} {self.symbol} @ ${current_price:.4f}")
            
            # Place order
            order = self.exchange.place_order(
                category="linear",
                symbol=self.linear,
                side=side,
                orderType="Market",
                qty=qty
            )
            
            if order.get('retCode') != 0:
                print(f"❌ Order failed: {order.get('retMsg')}")
                return False
            
            print(f"✅ Order placed: {order['result']['orderId']}")
            
            # Wait for position
            await asyncio.sleep(2)
            
            # Set stops (only initial stop loss and take profit)
            if signal['action'] == 'BUY':
                sl = self.risk_manager.get_stop_loss(current_price, 'long')
                tp = self.risk_manager.get_take_profit(current_price, 'long')
            else:
                sl = self.risk_manager.get_stop_loss(current_price, 'short')
                tp = self.risk_manager.get_take_profit(current_price, 'short')
            
            stops_resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                stopLoss=self.format_price(info, sl),
                takeProfit=self.format_price(info, tp),
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice"
            )
            
            if stops_resp.get('retCode') == 0:
                print(f"✅ Stops set - SL: ${sl:.4f}, TP: ${tp:.4f}")
            else:
                print(f"⚠️  Failed to set stops: {stops_resp.get('retMsg')}")
            
            # Reset profit lock state
            self.profit_lock_active = False
            
            # Send notification
            await self.notifier.trade_opened(self.symbol, current_price, float(qty))
            
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def close_position(self):
        """Close current position"""
        try:
            if not self.position:
                return False
            
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = str(self.position['size'])
            
            print(f"\n📉 Closing position: {qty} {self.symbol}")
            
            # Place closing order
            order = self.exchange.place_order(
                category="linear",
                symbol=self.linear,
                side=side,
                orderType="Market",
                qty=qty,
                reduceOnly=True
            )
            
            if order.get('retCode') != 0:
                print(f"❌ Close failed: {order.get('retMsg')}")
                return False
            
            print(f"✅ Position closed")
            
            # Calculate P&L
            await asyncio.sleep(2)
            
            # Send notification
            pnl = self.position.get('unrealized_pnl', 0)
            pnl_pct = self.position.get('unrealized_pnl_pct', 0)
            await self.notifier.trade_closed(self.symbol, pnl_pct, pnl)
            
            # Reset states
            self.position = None
            self.profit_lock_active = False
            return True
            
        except Exception as e:
            print(f"❌ Position close error: {e}")
            return False

    async def run_cycle(self):
        """Run one trading cycle"""
        try:
            # Get market data
            df = self.get_market_data()
            if df is None:
                return
            
            # Check position
            self.check_position()
            
            # Generate signal and get indicators
            signal = self.strategy.generate_signal(df)
            
            # Get current values
            current_price = df['close'].iloc[-1]
            
            # Calculate RSI and MFI directly
            try:
                # RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
                
                # MFI calculation
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                money_flow = typical_price * df['volume']
                delta_tp = typical_price.diff()
                positive_flow = money_flow.where(delta_tp > 0, 0).rolling(window=14).sum()
                negative_flow = money_flow.where(delta_tp < 0, 0).rolling(window=14).sum()
                money_ratio = positive_flow / negative_flow
                current_mfi = 100 - (100 / (1 + money_ratio.iloc[-1]))
                
            except Exception as e:
                current_rsi = 0.0
                current_mfi = 0.0
            
            # Check profit lock if we have a position
            if self.position:
                await self.check_profit_lock(current_price)
            
            # Log current state
            position_info = 'None'
            if self.position:
                pnl_pct = self.position['unrealized_pnl_pct']
                lock_status = '🔒' if self.profit_lock_active else ''
                position_info = f"{self.position['side']} ({pnl_pct:+.2f}%) {lock_status}"
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"Price: ${current_price:.4f} | "
                f"RSI: {current_rsi:.1f} | "
                f"MFI: {current_mfi:.1f} | "
                f"Position: {position_info}", 
                end='', flush=True)
            
            # Handle signals (only if not in profit lock mode)
            if signal and not self.profit_lock_active:
                print(f"\n🎯 Signal: {signal['action']} @ ${signal['price']:.4f}")
                
                if self.position:
                    # Check if we should reverse position
                    if (self.position['side'] == 'Buy' and signal['action'] == 'SELL') or \
                    (self.position['side'] == 'Sell' and signal['action'] == 'BUY'):
                        await self.close_position()
                        await asyncio.sleep(2)
                        await self.open_position(signal)
                else:
                    # Open new position
                    await self.open_position(signal)
                
                self.last_signal_time = datetime.now()
            
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")

    
    async def run(self):
        """Main trading loop"""
        print("\n🤖 RSI+MFI Trading Bot Starting...")
        
        # Connect to exchange
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        # Get initial balance
        balance = self.get_account_balance()
        print(f"💰 Account balance: ${balance:.2f}")
        print(f"📊 Trading {self.symbol}")
        print(f"⏱️  Timeframe: {self.timeframe} minutes")
        print(f"🎯 Strategy: RSI+MFI")
        print(f"🔒 Profit Lock: {self.break_even_pct}% | Trailing: {self.trailing_stop_distance}%")
        print("\n" + "="*60 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(30)  # Run every 30 seconds
                
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
    
    async def stop(self):
        """Stop the trading engine"""
        print("\n⚠️  Stopping trading engine...")
        self.running = False
        
        # Close any open position
        if self.position:
            print("Closing open position...")
            await self.close_position()
        
        print("✅ Trading engine stopped")