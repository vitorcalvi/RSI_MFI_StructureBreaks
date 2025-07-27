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
from core.telegram_notifier import TelegramNotifier


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
                        'unrealized_pnl': float(position['unrealisedPnl'])
                    }
                    return self.position
            self.position = None
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
            
            # Set stops
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
            pnl_pct = (pnl / (self.position['avg_price'] * self.position['size'])) * 100
            await self.notifier.trade_closed(self.symbol, pnl_pct, pnl)
            
            self.position = None
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
            
            # Log current state
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"Price: ${current_price:.4f} | "
                f"RSI: {current_rsi:.1f} | "
                f"MFI: {current_mfi:.1f} | "
                f"Position: {self.position['side'] if self.position else 'None'}", 
                end='', flush=True)
            
            # Handle signals
            if signal:
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