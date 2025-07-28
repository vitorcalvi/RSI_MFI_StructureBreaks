import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv(override=True)

from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
from core.risk_management import RiskManager
from core.telegram_notifier import TelegramNotifier

class TradeEngine:
    def __init__(self):
        # Components
        self.risk_manager = RiskManager()
        self.strategy = RSIMFICloudStrategy()
        self.notifier = TelegramNotifier()
        
        # Configuration
        self.symbol = self.risk_manager.symbol
        self.linear = self.symbol.replace('/', '')
        self.timeframe = '5'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API credentials
        if self.demo_mode:
            self.api_key = os.getenv('TESTNET_BYBIT_API_KEY')
            self.api_secret = os.getenv('TESTNET_BYBIT_API_SECRET')
        else:
            self.api_key = os.getenv('LIVE_BYBIT_API_KEY')
            self.api_secret = os.getenv('LIVE_BYBIT_API_SECRET')
        
        # State variables
        self.exchange = None
        self.running = False
        self.position = None
        self.profit_lock_active = False
        self.reversal_cooldown_cycles = 0
        self.last_trailing_update = None
        
    def connect(self):
        """Initialize exchange connection"""
        try:
            print("✅ Connecting to Bybit...")
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            server_time = self.exchange.get_server_time()
            if server_time.get('retCode') == 0:
                print(f"✅ Connected to Bybit {'Testnet' if self.demo_mode else 'Live'}")
                self.set_leverage()
                return True
            else:
                print(f"❌ Connection failed: {server_time.get('retMsg')}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def set_leverage(self):
        """Set leverage with error handling"""
        try:
            print(f"🔧 Setting leverage to {self.risk_manager.leverage}x...")
            resp = self.exchange.set_leverage(
                category="linear",
                symbol=self.linear,
                buyLeverage=str(self.risk_manager.leverage),
                sellLeverage=str(self.risk_manager.leverage)
            )
            
            ret_code = resp.get('retCode', 0)
            if ret_code in [0, 110043, 110036]:
                print(f"✅ Leverage set to {self.risk_manager.leverage}x")
            else:
                print(f"ℹ️ Leverage: {resp.get('retMsg')} (continuing)")
        except Exception as e:
            print(f"ℹ️ Leverage setting skipped: {str(e)[:50]}... (continuing)")
    
    def get_market_data(self):
        """Fetch market data"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.linear,
                interval=self.timeframe,
                limit=100
            )
            
            if klines.get('retCode') != 0:
                print(f"❌ API error: {klines.get('retMsg')}")
                return None
            
            # Check if result structure exists
            if not klines.get('result') or not klines['result'].get('list'):
                print("❌ No market data in response")
                return None
                
            data = klines['result']['list']
            if not data:
                print("❌ Empty market data")
                return None
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df = df.sort_index()
            
            # Ensure we have data
            if df.empty:
                print("❌ Empty DataFrame after processing")
                return None
                
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
                position_size = float(position['size'])
                
                if position_size > 0:
                    avg_price = float(position['avgPrice'])
                    unrealized_pnl = float(position['unrealisedPnl'])
                    investment = (avg_price * position_size) / self.risk_manager.leverage
                    pnl_pct = (unrealized_pnl / investment) * 100
                    
                    self.position = {
                        'side': position['side'],
                        'size': position_size,
                        'avg_price': avg_price,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': pnl_pct
                    }
                    return self.position
            
            # No position
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
        """Format quantity"""
        step = info['qty_step']
        qty = float(int(raw_qty / step) * step)
        qty = max(qty, info['min_qty'])
        
        step_str = f"{step:g}"
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        return f"{qty:.{decimals}f}" if decimals else str(int(qty))
    
    def format_price(self, info, price):
        """Format price"""
        tick = info['tick_size']
        price = round(price / tick) * tick
        
        tick_str = f"{tick:.20f}".rstrip('0').rstrip('.')
        decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{decimals}f}"

    async def check_loss_switch(self):
        """Switch position when loss threshold reached"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        threshold_pct = self.risk_manager.loss_switch_threshold * 100
        
        if pnl_pct <= threshold_pct:
            current_side = self.position['side']
            new_side = 'SELL' if current_side == 'Buy' else 'BUY'
            
            print(f"\n⚠️ Loss threshold reached: {pnl_pct:.2f}%")
            print(f"🔄 Switching to {new_side}")
            
            await self.close_position("Loss Limit")
            await asyncio.sleep(2)
            
            # Create switch signal
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            switch_signal = {
                'action': new_side,
                'price': current_price,
                'rsi': 50,
                'mfi': 50,
                'trend': 'LOSS_SWITCH',
                'timestamp': pd.Timestamp.now(),
                'confidence': 'LOSS_SWITCH'
            }
            
            success = await self.open_position(switch_signal)
            if success:
                await self.notifier.position_switched(
                    self.symbol, current_side, new_side,
                    self.position['size'], pnl_pct, self.position['unrealized_pnl']
                )

    async def check_profit_lock(self, current_price):
        """Check if profit lock should be activated"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        threshold_pct = self.risk_manager.break_even_pct * 100 * self.risk_manager.leverage
        
        # Debug: Show profit lock progress when profitable
        if pnl_pct > 0 and not self.profit_lock_active:
            print(f" [Lock: {pnl_pct:.1f}%/{threshold_pct:.0f}%]", end='')
        
        if not self.profit_lock_active and pnl_pct >= threshold_pct:
            self.profit_lock_active = True
            print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}% (threshold: {threshold_pct:.1f}%)")
            
            await self.notifier.profit_lock_activated(
                self.symbol, pnl_pct, self.risk_manager.trailing_stop_distance * 100
            )
            
            # Set trailing stop with CORRECT implementation
            info = self.get_symbol_info()
            if info:
                # Calculate absolute trailing distance (not percentage)
                trailing_distance = current_price * self.risk_manager.trailing_stop_distance
                formatted_trailing_distance = self.format_price(info, trailing_distance)
                
                resp = self.exchange.set_trading_stop(
                    category="linear",
                    symbol=self.linear,
                    positionIdx=0,
                    takeProfit="",
                    stopLoss="",  # Clear existing SL
                    trailingStop=formatted_trailing_distance,  # Absolute price distance
                    activePrice=self.format_price(info, current_price)  # Activate at current price
                )
                
                if resp.get('retCode') == 0:
                    print(f"✅ Trailing stop set: {formatted_trailing_distance} distance")
                else:
                    print(f"❌ Trailing stop failed: {resp.get('retMsg')}")
            
            self.last_trailing_update = datetime.now()

    async def open_position(self, signal):
        """Open position"""
        try:
            balance = self.get_account_balance()
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            position_size = self.risk_manager.calculate_position_size(balance, current_price)
            info = self.get_symbol_info()
            if not info:
                return False
            
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            print(f"\n📈 Opening {side}: {qty} @ ${current_price:.4f}")
            
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
            
            await asyncio.sleep(2)
            
            # Set initial stops (TP/SL only, no trailing stop initially)
            try:
                if signal['action'] == 'BUY':
                    sl = self.risk_manager.get_stop_loss(current_price, 'long')
                    tp = self.risk_manager.get_take_profit(current_price, 'long')
                else:
                    sl = self.risk_manager.get_stop_loss(current_price, 'short')
                    tp = self.risk_manager.get_take_profit(current_price, 'short')
                
                stop_resp = self.exchange.set_trading_stop(
                    category="linear",
                    symbol=self.linear,
                    positionIdx=0,
                    stopLoss=self.format_price(info, sl),
                    takeProfit=self.format_price(info, tp),
                    slTriggerBy="LastPrice",
                    tpTriggerBy="LastPrice"
                )
                
                if stop_resp.get('retCode') == 0:
                    print(f"✅ Initial stops set - SL: ${sl:.6f}, TP: ${tp:.6f}")
                else:
                    print(f"⚠️ Stop setting failed: {stop_resp.get('retMsg')} (continuing)")
                    
            except Exception as e:
                print(f"⚠️ Stop setting error: {e} (continuing)")
            
            self.profit_lock_active = False
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def close_position(self, reason="Signal"):
        """Close position"""
        try:
            if not self.position:
                return False
            
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = str(self.position['size'])
            
            print(f"\n📉 Closing position (Reason: {reason})")
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.linear,
                side=side,
                orderType="Market",
                qty=qty,
                reduceOnly=True
            )
            
            if order.get('retCode') != 0:
                return False
            
            pnl = self.position.get('unrealized_pnl', 0)
            pnl_pct = self.position.get('unrealized_pnl_pct', 0)
            
            if reason == "Profit Protection" and pnl_pct >= self.risk_manager.profit_protection_threshold:
                self.reversal_cooldown_cycles = 3
            
            await self.notifier.trade_closed(self.symbol, pnl_pct, pnl, reason)
            
            self.position = None
            self.profit_lock_active = False
            return True
            
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False

    async def run_cycle(self):
        """Run trading cycle"""
        try:
            df = self.get_market_data()
            if df is None or df.empty:
                print("⚠️ No market data available")
                return
            
            self.check_position()
            signal = self.strategy.generate_signal(df)
            
            # Safe access to current price
            try:
                current_price = df['close'].iloc[-1]
            except (IndexError, KeyError):
                print("⚠️ Cannot get current price")
                return
            
            # Get indicators from strategy (avoid duplicate calculation)
            current_rsi = current_mfi = 50.0
            current_trend = "UNKNOWN"
            
            if len(df) >= 26:
                try:
                    df_with_indicators = self.strategy.calculate_indicators(df)
                    if df_with_indicators is not None and not df_with_indicators.empty:
                        current_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50.0
                        current_mfi = df_with_indicators['mfi'].iloc[-1] if 'mfi' in df_with_indicators.columns else 50.0
                        current_trend = df_with_indicators['trend'].iloc[-1] if 'trend' in df_with_indicators.columns else "UNKNOWN"
                except Exception as e:
                    print(f"⚠️ Indicator calculation error: {e}")
            
            # Handle cooldown
            if self.reversal_cooldown_cycles > 0:
                self.reversal_cooldown_cycles -= 1
                if self.reversal_cooldown_cycles == 0:
                    print(f"\n🔓 Cooldown ended")
            
            # Check position states
            if self.position:
                await self.check_profit_lock(current_price)
                await self.check_loss_switch()
            
            # Display status
            position_info = 'None'
            cooldown_info = ''
            trend_emoji = {"UP": "🟢", "DOWN": "🔴", "SIDEWAYS": "🟡", "UNKNOWN": "⚪"}.get(current_trend, "⚪")
            
            if self.reversal_cooldown_cycles > 0:
                cooldown_info = f' [Cooldown: {self.reversal_cooldown_cycles}]'
            if self.position:
                pnl_pct = self.position['unrealized_pnl_pct']
                lock_status = '🔒' if self.profit_lock_active else ''
                position_info = f"{self.position['side']} ({pnl_pct:+.2f}%) {lock_status}"
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                f"{trend_emoji}{current_trend} | {position_info}{cooldown_info}", 
                end='', flush=True)
            
            # Handle signals
            if signal:
                await self.handle_signal(signal)
                
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
    
    async def handle_signal(self, signal):
        """Handle trading signal"""
        signal_action = signal['action']
        
        if self.position:
            current_side = self.position['side']
            pnl_pct = self.position['unrealized_pnl_pct']
            
            should_reverse = (
                (current_side == 'Buy' and signal_action == 'SELL') or 
                (current_side == 'Sell' and signal_action == 'BUY')
            )
            
            if should_reverse:
                if pnl_pct >= self.risk_manager.profit_protection_threshold:
                    print(f"\n💰 Taking +{pnl_pct:.2f}% profit")
                    await self.close_position("Profit Protection")
                elif pnl_pct <= -5.0:
                    print(f"\n🔄 Reversing losing position: {pnl_pct:.2f}%")
                    await self.close_position("Reverse Signal")
                    await asyncio.sleep(2)
                    await self.open_position(signal)
                else:
                    print(f"\n⏸️ Signal ignored - P&L: {pnl_pct:.2f}%")
        else:
            if self.reversal_cooldown_cycles > 0:
                print(f"\n⏸️ Cooldown active ({self.reversal_cooldown_cycles} cycles)")
            else:
                print(f"\n🎯 Signal: {signal_action} @ ${signal['price']:.4f}")
                await self.open_position(signal)
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(10)
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
            await self.notifier.error_notification(str(e))
    
    async def stop(self):
        """Stop trading engine"""
        print("\n⚠️ Stopping...")
        self.running = False
        
        if self.position:
            await self.close_position("Bot Stop")
        
        await self.notifier.bot_stopped()
        print("✅ Stopped")