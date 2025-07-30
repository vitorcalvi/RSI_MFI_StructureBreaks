import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
from core.risk_management import RiskManager
from core.telegram_notifier import TelegramNotifier

load_dotenv(override=True)

class TradeEngine:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.strategy = RSIMFICloudStrategy(self.risk_manager)  # Pass risk manager
        self.notifier = TelegramNotifier()
        
        self.symbol = self.risk_manager.symbol  # Get from risk manager
        self.linear = self.risk_manager.linear
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API credentials
        if self.demo_mode:
            self.api_key = os.getenv('TESTNET_BYBIT_API_KEY')
            self.api_secret = os.getenv('TESTNET_BYBIT_API_SECRET')
        else:
            self.api_key = os.getenv('LIVE_BYBIT_API_KEY')
            self.api_secret = os.getenv('LIVE_BYBIT_API_SECRET')
        
        # State
        self.exchange = None
        self.running = False
        self.position = None
        self.profit_lock_active = False
        self.entry_price = 0
        self.position_side = None
    
    def connect(self):
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            server_time = self.exchange.get_server_time()
            if server_time.get('retCode') == 0:
                mode = "Testnet" if self.demo_mode else "Live"
                print(f"✅ Connected to Bybit {mode}")
                return True
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.linear,
                interval='5',
                limit=100
            )
            
            if klines.get('retCode') != 0 or not klines.get('result', {}).get('list'):
                return None
            
            data = klines['result']['list']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df.sort_index()
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return None
    
    def get_wallet_balance(self):
        try:
            resp = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if resp.get('retCode') == 0:
                for coin in resp['result']['list'][0].get('coin', []):
                    if coin.get('coin') == 'USDT':
                        return float(coin.get('walletBalance', 0))
            return 0
        except:
            return 0

    def check_position(self):
        """Simple position check"""
        try:
            pos_resp = self.exchange.get_positions(category="linear", symbol=self.linear)
            if pos_resp.get('retCode') != 0:
                self.position = None
                return None
                
            positions = pos_resp.get('result', {}).get('list', [])
            if not positions:
                self.position = None
                return None
            
            position = positions[0]
            position_size = float(position.get('size', 0))
            
            if position_size <= 0:
                self.position = None
                return None
            
            # We have a position
            self.position = {
                'side': position.get('side'),
                'size': position_size,
                'avg_price': float(position.get('avgPrice', 0)),
                'unrealized_pnl': float(position.get('unrealisedPnl', 0))
            }
            
            self.entry_price = self.position['avg_price']
            self.position_side = self.position['side'].lower()
            
            return self.position
            
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            return None

    def get_symbol_info(self):
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
        except:
            return None
    
    def format_qty(self, info, raw_qty):
        step = info['qty_step']
        qty = float(int(raw_qty / step) * step)
        qty = max(qty, info['min_qty'])
        
        step_str = f"{step:g}"
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        return f"{qty:.{decimals}f}" if decimals else str(int(qty))
    
    def format_price(self, info, price):
        tick = info['tick_size']
        if tick == 0:
            return str(price)
        price = round(price / tick) * tick
        
        tick_str = f"{tick:.20f}".rstrip('0').rstrip('.')
        decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{decimals}f}"

    async def handle_risk_management(self, current_price):
        """Simple risk management"""
        if not self.position or not self.entry_price:
            return
            
        # Check for profit lock activation
        if not self.profit_lock_active:
            if self.risk_manager.should_activate_profit_lock(
                self.entry_price, current_price, self.position_side):
                
                self.profit_lock_active = True
                print(f"\n🔒 PROFIT LOCK ACTIVATED")
                
                # Set trailing stop
                await self._set_trailing_stop(current_price)
                
                await self.notifier.profit_lock_activated(
                    self.symbol, 0, self.risk_manager.trailing_stop_pct * 100
                )

    async def _set_trailing_stop(self, current_price):
        """Set trailing stop"""
        try:
            info = self.get_symbol_info()
            if not info:
                return
                
            trailing_price = self.risk_manager.get_trailing_stop_price(
                current_price, self.position_side)
            
            trailing_distance = abs(current_price - trailing_price)
            formatted_trailing = self.format_price(info, trailing_distance)
            
            resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                trailingStop=formatted_trailing
            )
            
            if resp.get('retCode') == 0:
                print(f"✅ Trailing stop: ${formatted_trailing}")
            else:
                print(f"⚠️ Trailing stop failed: {resp.get('retMsg')}")
                
        except Exception as e:
            print(f"⚠️ Trailing stop error: {e}")

    async def open_position(self, signal):
        """Open position with simple risk management"""
        try:
            # Close existing position first
            if self.position:
                await self.close_position("Force Close")
                await asyncio.sleep(2)
                
                self.check_position()
                if self.position:
                    print(f"❌ Could not close existing position")
                    return False
            
            wallet_balance = self.get_wallet_balance()
            current_price = signal['price']
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(wallet_balance, current_price)
            
            info = self.get_symbol_info()
            if not info:
                return False
            
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            print(f"\n📈 Opening {side}: {qty} @ ${current_price:.4f}")
            print(f"💰 Risk: ${self.risk_manager.fixed_risk_usd:.0f}")
            
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
            
            # Set stop loss and take profit
            await self._set_stop_and_tp(signal, current_price, info)
            
            self.profit_lock_active = False
            self.entry_price = current_price
            self.position_side = side.lower()
            
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def _set_stop_and_tp(self, signal, current_price, info):
        """Set stop loss and take profit"""
        try:
            side = 'long' if signal['action'] == 'BUY' else 'short'
            
            sl_price = self.risk_manager.get_stop_loss_price(current_price, side)
            tp_price = self.risk_manager.get_take_profit_price(current_price, side)
            
            # Set both SL and TP
            stop_resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                stopLoss=self.format_price(info, sl_price),
                takeProfit=self.format_price(info, tp_price),
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice"
            )
            
            if stop_resp.get('retCode') == 0:
                print(f"✅ SL: ${sl_price:.4f} | TP: ${tp_price:.4f}")
            else:
                print(f"⚠️ SL/TP failed: {stop_resp.get('retMsg')}")
            
        except Exception as e:
            print(f"⚠️ SL/TP error: {e}")
    
    async def close_position(self, reason="Signal"):
        """Close position"""
        try:
            if not self.position:
                return False
            
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = str(self.position['size'])
            
            print(f"\n📉 Closing position ({reason})")
            
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
            
            pnl = self.position.get('unrealized_pnl', 0)
            await self.notifier.trade_closed(self.symbol, 0, pnl, reason)
            
            self.position = None
            self.profit_lock_active = False
            self.entry_price = 0
            self.position_side = None
            
            return True
            
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False

    async def handle_signal(self, signal):
        """Handle trading signals"""
        if not signal:
            return
        
        if self.position:
            # Close on opposite signal
            current_side = self.position['side']
            is_opposite = (
                (current_side == 'Buy' and signal['action'] == 'SELL') or 
                (current_side == 'Sell' and signal['action'] == 'BUY')
            )
            
            if is_opposite:
                await self.close_position("Opposite Signal")
        else:
            # Open new position
            print(f"\n🎯 Signal: {signal['action']} @ ${signal['price']:.4f}")
            await self.open_position(signal)

    async def run_cycle(self):
        try:
            # Get data
            df = self.get_market_data()
            if df is None or df.empty:
                return
            
            # Check position
            self.check_position()
            
            # Get signal and current price
            signal = self.strategy.generate_signal(df)
            current_price = df['close'].iloc[-1]
            
            # Risk management
            if self.position:
                await self.handle_risk_management(current_price)
                self.check_position()
            
            # Display status
            self._display_status(df, current_price)
            
            # Handle signals
            await self.handle_signal(signal)
                
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
    
    def _display_status(self, df, current_price):
        """Simple status display"""
        # Get indicators
        df_with_indicators = self.strategy.calculate_indicators(df)
        current_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50.0
        current_mfi = df_with_indicators['mfi'].iloc[-1] if 'mfi' in df_with_indicators.columns else 50.0
        
        # Position info
        position_info = 'None'
        if self.position:
            pnl = self.position['unrealized_pnl']
            lock_status = '🔒' if self.profit_lock_active else ''
            position_info = f"{self.position['side']} (${pnl:+.2f}) {lock_status}"
        
        status = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                  f"Pos:{position_info}")
        
        print(f"\r{status}", end='', flush=True)
    
    async def run(self):
        self.running = True
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(1)
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
            await self.notifier.error_notification(str(e))
    
    async def stop(self):
        print("\n⚠️ Stopping...")
        self.running = False
        
        if self.position:
            await self.close_position("Bot Stop")
        
        print("✅ Stopped")