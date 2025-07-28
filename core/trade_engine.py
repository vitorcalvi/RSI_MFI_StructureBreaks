import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

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


class TradeEngine:
    def __init__(self):
        # Configuration
        self.symbol = os.getenv('SYMBOLS')
        self.linear = self.symbol.replace('/', '')
        self.timeframe = '5'  # 5 minutes
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.leverage = int(os.getenv('LEVERAGE'))  # Set leverage from env or default to 10x
        
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
        
        # Profit Locker - Fixed minimum threshold
        self.break_even_pct = 0.5  # 0.5% minimum for Bybit trailing stop
        self.profit_lock_active = False
        self.trailing_stop_distance = 1.0  # 1.0% trailing stop distance (minimum for Bybit)
        
        # Position switching for losses
        self.loss_switch_threshold = -1.2  # Switch position at -2% loss
        
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
                # Set leverage after connection
                self.set_leverage()
                return True
            else:
                print(f"❌ Connection failed: {server_time.get('retMsg')}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def set_leverage(self):
        """Set leverage for the trading symbol"""
        try:
            print(f"🔧 Setting leverage to {self.leverage}x...")
            resp = self.exchange.set_leverage(
                category="linear",
                symbol=self.linear,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage)
            )
            
            if resp.get('retCode') == 0:
                print(f"✅ Leverage set to {self.leverage}x")
            else:
                print(f"❌ Failed to set leverage: {resp.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Leverage setting error: {e}")
    
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
        """Check current position - Fixed to handle closed positions"""
        try:
            pos_resp = self.exchange.get_positions(category="linear", symbol=self.linear)
            if pos_resp.get('retCode') == 0 and pos_resp['result']['list']:
                position = pos_resp['result']['list'][0]
                position_size = float(position['size'])
                
                # Check if position exists and has size > 0
                if position_size > 0:
                    self.position = {
                        'side': position['side'],
                        'size': position_size,
                        'avg_price': float(position['avgPrice']),
                        'unrealized_pnl': float(position['unrealisedPnl']),
                        'unrealized_pnl_pct': float(position['unrealisedPnl']) / (float(position['avgPrice']) * position_size) * 100
                    }
                    return self.position
                else:
                    # Position was closed
                    if self.position is not None:
                        print(f"\n📉 Position closed by system")
                    self.position = None
                    self.profit_lock_active = False
                    return None
            
            # No position data
            if self.position is not None:
                print(f"\n📉 Position closed")
            self.position = None
            self.profit_lock_active = False
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

    async def switch_position_mode(self):
        """Switch position using Bybit's position mode API"""
        try:
            if not self.position:
                return False
                
            current_side = self.position['side']
            size = self.position['size']
            
            # Determine new side
            new_side = "Sell" if current_side == "Buy" else "Buy"
            
            print(f"\n🔄 Switching position: {current_side} -> {new_side}")
            
            # Use Bybit's position switch API
            switch_resp = self.exchange.switch_position_mode(
                category="linear",
                symbol=self.linear,
                coin="USDT",
                mode=1  # Switch to opposite side
            )
            
            if switch_resp.get('retCode') == 0:
                print(f"✅ Position switched successfully")
                
                # Wait for position update
                await asyncio.sleep(2)
                
                return True
            else:
                print(f"❌ Position switch failed: {switch_resp.get('retMsg')}")
                return False
                
        except Exception as e:
            print(f"❌ Position switch error: {e}")
            return False

    async def check_loss_switch(self):
        """Check if we should switch position due to losses"""
        if not self.position or self.profit_lock_active:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        
        # Check if loss threshold is reached
        if pnl_pct <= self.loss_switch_threshold:
            print(f"\n⚠️ Loss threshold reached: {pnl_pct:.2f}%")
            
            current_side = self.position['side']
            size = self.position['size']
            
            # Try to switch position
            switch_success = await self.switch_position_mode()
            
            if not switch_success:
                # Fallback: close position if switch fails
                print("🚨 Switch failed, closing position")
                await self.close_position("Loss Limit")
            else:
                # Notify about successful switch
                await self.notifier.position_switched(
                    self.symbol,
                    current_side,
                    "Sell" if current_side == "Buy" else "Buy", 
                    size,
                    pnl_pct,
                    self.position.get('unrealized_pnl', 0)
                )


    async def update_trailing_stop(self, current_price):
        """Update trailing stop loss - FIXED for Bybit 10% minimum requirement"""
        try:
            if not self.position or not self.profit_lock_active:
                return False
                
            side = self.position['side']
            
            # FIXED: Bybit requires minimum 10% trailing distance
            distance_pct = 0.10  # 10% minimum trailing distance
            
            if side == 'Buy':
                # For long: trailing stop below current price
                trailing_stop_price = current_price * (1 - distance_pct)
                active_price = current_price * 1.001  # Just above current
            else:
                # For short: trailing stop above current price  
                trailing_stop_price = current_price * (1 + distance_pct)
                active_price = current_price * 0.999  # Just below current
            
            info = self.get_symbol_info()
            if not info:
                return False
                
            formatted_trailing_stop = self.format_price(info, trailing_stop_price)
            formatted_active_price = self.format_price(info, active_price)
            
            # Update trailing stop
            resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                trailingStop=formatted_trailing_stop,
                activePrice=formatted_active_price
            )
            
            if resp.get('retCode') == 0:
                return True
            else:
                return False
                
        except Exception as e:
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
                10.0  # FIXED: Use 10% for notification
            )
            
            # Remove take profit and set trailing stop
            try:
                info = self.get_symbol_info()
                if info:
                    side = self.position['side']
                    distance_pct = 0.10  # FIXED: 10% minimum distance
                    
                    if side == 'Buy':
                        trailing_stop_price = current_price * (1 - distance_pct)
                        active_price = current_price * 1.005  # 0.5% above current
                    else:
                        trailing_stop_price = current_price * (1 + distance_pct)
                        active_price = current_price * 0.995  # 0.5% below current
                    
                    formatted_trailing_stop = self.format_price(info, trailing_stop_price)
                    formatted_active_price = self.format_price(info, active_price)
                    
                    print(f"Setting trailing stop: {formatted_trailing_stop}, active: {formatted_active_price}")
                    
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
            
            # Send notification with side parameter
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def close_position(self, reason="Signal"):
        """Close current position"""
        try:
            if not self.position:
                return False
            
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = str(self.position['size'])
            
            print(f"\n📉 Closing position: {qty} {self.symbol} (Reason: {reason})")
            
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
            pnl = self.position.get('unrealized_pnl', 0)
            pnl_pct = self.position.get('unrealized_pnl_pct', 0)
            
            # Send notification with reason
            await self.notifier.trade_closed(self.symbol, pnl_pct, pnl, reason)
            
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
            
            # Check position status first
            old_position = self.position
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
                await self.check_loss_switch()  # Check for loss switching
            
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
            
            # Handle signals 
            if signal:
                signal_action = signal['action']
                
                # Check if we have a position
                if self.position:
                    current_side = self.position['side']
                    
                    # Check for opposite signal (position reversal)
                    should_reverse = (
                        (current_side == 'Buy' and signal_action == 'SELL') or 
                        (current_side == 'Sell' and signal_action == 'BUY')
                    )
                    
                    if should_reverse:
                        print(f"\n🔄 Reversing position: {current_side} -> {signal_action}")
                        await self.close_position("Reverse Signal")
                        await asyncio.sleep(2)
                        await self.open_position(signal)
                    # If same direction signal, ignore (already in position)
                    
                else:
                    # No position, open new one
                    print(f"\n🎯 Signal: {signal_action} @ ${signal['price']:.4f}")
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
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"⏱️  Timeframe: {self.timeframe} minutes")
        print(f"🎯 Strategy: RSI+MFI")
        print(f"🔒 Profit Lock: {self.break_even_pct}% | Trailing: {self.trailing_stop_distance}%")
        print(f"🔄 Loss Switch: {self.loss_switch_threshold}%")
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
            await self.close_position("Bot Stop")
        
        print("✅ Trading engine stopped")