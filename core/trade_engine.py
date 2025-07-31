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
        self.strategy = RSIMFICloudStrategy(self.risk_manager)
        
        print("✅ Risk management initialized")
        print("✅ RSI/MFI strategy loaded")
        print("🎯 Break & Retest pattern detection active")
        
        self.notifier = TelegramNotifier()
        self.symbol = self.risk_manager.symbol
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
        self.position_start_time = None
        self.pending_order = None
        
        # Error tracking
        self._last_error = {}
    
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
            print(f"❌ Connection Error | {e}")
            return False
    
    def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.linear,
                interval='5',
                limit=200
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
            self._log_error('market_data', e)
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
        try:
            pos_resp = self.exchange.get_positions(category="linear", symbol=self.linear)
            if pos_resp.get('retCode') != 0:
                self._clear_position()
                return None
                
            positions = pos_resp.get('result', {}).get('list', [])
            if not positions:
                self._clear_position()
                return None
            
            position = positions[0]
            position_size = float(position.get('size', 0))
            
            if position_size <= 0:
                self._clear_position()
                return None
            
            if not self.position:
                self.position_start_time = datetime.now()
            
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
            self._log_error('position', e)
            self._clear_position()
            return None
    
    def _clear_position(self):
        self.position = None
        self.profit_lock_active = False
        self.entry_price = 0
        self.position_side = None
        self.position_start_time = None
        self.pending_order = None

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

    async def check_structure_breaks(self, df, current_price):
        if not self.position or not self.entry_price:
            return None
        
        break_signal = self.strategy.detect_structure_break(
            df, self.position_side, self.entry_price, current_price)
        
        if break_signal:
            print(f"\n🚨 STRUCTURE BREAK | {break_signal['break_type']} | Price: ${current_price:.2f}")
            
            close_success = await self.close_position(f"Structure Break")
            
            if close_success:
                await asyncio.sleep(1)
                flip_signal = {
                    'action': break_signal['flip_signal'],
                    'price': current_price,
                    'rsi': 50.0,
                    'mfi': 50.0,
                    'timestamp': df.index[-1],
                    'structure_stop': self.strategy.get_structure_stop(df, break_signal['flip_signal'], current_price),
                    'signal_type': 'STRUCTURE_FLIP'
                }
                
                print(f"🔄 FLIPPING | {break_signal['flip_signal']}")
                await self.open_position(flip_signal)
            
            return break_signal
        
        return None

    async def handle_risk_management(self, current_price):
        if not self.position or not self.entry_price:
            return
            
        if not self.profit_lock_active:
            if self.risk_manager.should_activate_profit_lock(
                self.entry_price, current_price, self.position_side):
                
                self.profit_lock_active = True
                
                if self.position_side == 'buy':
                    profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                else:
                    profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                
                trail_pct = getattr(self.risk_manager, 'trailing_stop_pct', 0.01) * 100
                print(f"\n🔒 Profit Lock | +{profit_pct:.1f}% | Trailing: {trail_pct:.1f}%")
                
                await self._set_trailing_stop(current_price)
                await self.notifier.profit_lock_activated(self.symbol, profit_pct, trail_pct)

    async def _set_trailing_stop(self, current_price):
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
            
            if resp.get('retCode') != 0:
                print(f"\n⚠️ Trailing Failed | {resp.get('retMsg', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n⚠️ Trailing Error | {e}")

    async def open_position(self, signal):
        try:
            if self.position:
                print(f"\n🔄 Force Close | Clearing position")
                await self.close_position("Force Close")
                await asyncio.sleep(2)
                
                self.check_position()
                if self.position:
                    print(f"❌ Force Close Failed")
                    return False
            
            wallet_balance = self.get_wallet_balance()
            current_price = signal['price']
            structure_stop = signal.get('structure_stop')
            
            position_size = self.risk_manager.calculate_position_size(
                wallet_balance, current_price, structure_stop, signal.get('signal_type'))
            
            info = self.get_symbol_info()
            if not info:
                return False
            
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            # Calculate values for display
            side_type = 'long' if signal['action'] == 'BUY' else 'short'
            sl_price = self.risk_manager.get_stop_loss_price(
                current_price, side_type, structure_stop)
            tp_price = self.risk_manager.get_take_profit_price(
                current_price, side_type, structure_stop, signal.get('signal_type'))
            
            actual_risk = float(qty) * abs(current_price - sl_price)
            actual_reward = float(qty) * abs(tp_price - current_price)
            
            # Display signal
            signal_type_display = signal.get('signal_type', 'RSI_MFI')
            if signal_type_display == 'BREAK_RETEST':
                strength = signal.get('retest_strength', 0.6)
                print(f"\n🎯 BREAK RETEST | Strength: {strength:.1%}")
            
            self.pending_order = {
                'action': signal['action'],
                'qty': qty,
                'price': current_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'risk_amount': actual_risk,
                'reward_amount': actual_reward,
                'signal_type': signal_type_display
            }
            
            # Place order
            order = self.exchange.place_order(
                category="linear",
                symbol=self.linear,
                side=side,
                orderType="Market",
                qty=qty
            )
            
            if order.get('retCode') != 0:
                print(f"\n❌ Order Failed | {order.get('retMsg', 'Unknown error')}")
                self.pending_order = None
                return False
            
            # Set SL and TP
            await self._set_stop_and_tp(signal, current_price, info, structure_stop)
            
            self.profit_lock_active = False
            self.entry_price = current_price
            self.position_side = side.lower()
            self.position_start_time = datetime.now()
            
            await asyncio.sleep(1)
            self.check_position()
            
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"\n❌ Order Failed | {e}")
            self.pending_order = None
            return False

    async def _set_stop_and_tp(self, signal, current_price, info, structure_stop=None):
        try:
            side = 'long' if signal['action'] == 'BUY' else 'short'
            
            sl_price = self.risk_manager.get_stop_loss_price(
                current_price, side, structure_stop)
            tp_price = self.risk_manager.get_take_profit_price(
                current_price, side, structure_stop, signal.get('signal_type'))
            
            # Enhanced targets for retests
            if signal.get('signal_type') == 'BREAK_RETEST':
                risk_distance = abs(current_price - sl_price)
                extended_reward = risk_distance * 6
                
                if side == 'long':
                    tp_price = current_price + extended_reward
                else:
                    tp_price = current_price - extended_reward
                
                print(f"🎯 Extended Target | 6:1 R:R | TP: ${tp_price:.2f}")
            
            stop_resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                stopLoss=self.format_price(info, sl_price),
                takeProfit=self.format_price(info, tp_price),
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice"
            )
            
            if stop_resp.get('retCode') != 0:
                print(f"\n⚠️ SL/TP Warning | {stop_resp.get('retMsg', 'Unknown error')}")
            
        except Exception as e:
            print(f"\n⚠️ SL/TP Error | {e}")
    
    async def close_position(self, reason="Signal"):
        try:
            if not self.position:
                return False
            
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = str(self.position['size'])
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.linear,
                side=side,
                orderType="Market",
                qty=qty,
                reduceOnly=True
            )
            
            if order.get('retCode') != 0:
                print(f"\n❌ Close Failed | {order.get('retMsg', 'Unknown error')}")
                return False
            
            duration = "00:00:00"
            if self.position_start_time:
                time_diff = datetime.now() - self.position_start_time
                hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            pnl = self.position.get('unrealized_pnl', 0)
            result = "Win" if pnl > 0 else "Loss"
            
            print(f"\n📉 CLOSED | {reason} | ⏱️ {duration} | PnL: {pnl:+.2f} | {result}")
            
            await self.notifier.trade_closed(self.symbol, 0, pnl, reason)
            self._clear_position()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Close Failed | {e}")
            return False

    async def handle_signal(self, signal):
        if not signal:
            return
        
        # High priority for retest patterns
        if signal.get('signal_type') == 'BREAK_RETEST':
            print(f"\n🎯 HIGH PRIORITY | Break & Retest")
            
            if self.position:
                await self.close_position("Retest Priority")
                await asyncio.sleep(1)
            
            await self.open_position(signal)
            return
        
        # Regular signal handling
        if self.position:
            current_side = self.position['side']
            is_opposite = (
                (current_side == 'Buy' and signal['action'] == 'SELL') or 
                (current_side == 'Sell' and signal['action'] == 'BUY')
            )
            
            if is_opposite:
                await self.close_position("Opposite Signal")
        else:
            await self.open_position(signal)

    async def run_cycle(self):
        try:
            df = self.get_market_data()
            if df is None or df.empty:
                return
            
            self.check_position()
            current_price = df['close'].iloc[-1]
            
            # Check structure breaks
            if self.position:
                structure_break = await self.check_structure_breaks(df, current_price)
                if structure_break:
                    self._display_status(df, current_price)
                    return
            
            # Get signals
            signal = self.strategy.generate_signal(df)
            
            # Risk management
            if self.position:
                await self.handle_risk_management(current_price)
                self.check_position()
            
            # Display and handle
            self._display_status(df, current_price)
            await self.handle_signal(signal)
                
        except Exception as e:
            self._log_error('cycle', e)

    def _display_status(self, df, current_price):
        """Fixed display with None value handling"""
        try:
            df_with_indicators = self.strategy.calculate_indicators(df)
            
            # Safe RSI/MFI value extraction with None handling
            current_rsi = 50.0
            current_mfi = 50.0
            
            if 'rsi' in df_with_indicators.columns and len(df_with_indicators) > 0:
                rsi_val = df_with_indicators['rsi'].iloc[-1]
                if rsi_val is not None and not pd.isna(rsi_val):
                    current_rsi = float(rsi_val)
            
            if 'mfi' in df_with_indicators.columns and len(df_with_indicators) > 0:
                mfi_val = df_with_indicators['mfi'].iloc[-1]
                if mfi_val is not None and not pd.isna(mfi_val):
                    current_mfi = float(mfi_val)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            symbol_short = self.symbol.replace('/', '')
            
            # Retest monitoring status
            retest_status = ""
            if hasattr(self.strategy, 'retest_monitoring') and self.strategy.retest_monitoring:
                try:
                    break_info = self.strategy.retest_monitoring.get('break', {})
                    break_type = break_info.get('type', 'unknown')
                    break_level = break_info.get('level', 0)
                    if break_level > 0:
                        retest_status = f" | 🔍 {break_type} @ ${break_level:.2f}"
                except:
                    pass
            
            if self.pending_order:
                direction_emoji = "📈" if self.pending_order['action'] == 'BUY' else "📉"
                signal_type = self.pending_order.get('signal_type', 'RSI_MFI')
                
                pattern_info = ""
                if signal_type == 'BREAK_RETEST':
                    pattern_info = " (RETEST)"
                elif signal_type == 'STRUCTURE_FLIP':
                    pattern_info = " (FLIP)"
                
                opening_line = f"[{timestamp}] {symbol_short} | RSI: {current_rsi:.1f} | MFI: {current_mfi:.1f} | {direction_emoji} {self.pending_order['action']}{pattern_info} {self.pending_order['qty']} @ ${self.pending_order['price']:.2f}"
                print(f"\n{opening_line}")
                
                risk_line = (f"💰 Risk: ${self.pending_order['risk_amount']:.0f} | "
                            f"SL: ${self.pending_order['sl_price']:.2f} | "
                            f"TP: ${self.pending_order['tp_price']:.2f}")
                print(f"{risk_line}")
                
                self.pending_order = None
                
            elif self.position:
                if self.position_start_time:
                    duration = datetime.now() - self.position_start_time
                    hours, remainder = divmod(int(duration.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = "00:00:00"
                
                pnl = self.position.get('unrealized_pnl', 0)
                lock_status = ' 🔒' if self.profit_lock_active else ''
                
                monitor_line = f"⏱️ {duration_str} | ${current_price:.2f} | PnL: {pnl:+.2f}{lock_status}"
                print(f"\r{monitor_line}", end='', flush=True)
                
            else:
                status = f"[{timestamp}] {symbol_short} | RSI: {current_rsi:.1f} | MFI: {current_mfi:.1f} | No Position{retest_status}"
                print(f"\r{status}", end='', flush=True)
                
        except Exception as e:
            # Fallback display if anything goes wrong
            timestamp = datetime.now().strftime('%H:%M:%S')
            symbol_short = self.symbol.replace('/', '')
            status = f"[{timestamp}] {symbol_short} | Price: ${current_price:.2f} | Status: OK"
            print(f"\r{status}", end='', flush=True)
    
    def _log_error(self, error_type, error):
        now = datetime.now()
        if error_type not in self._last_error or (now - self._last_error[error_type]).seconds > 30:
            print(f"\n❌ {error_type.upper()} Error | {error}")
            self._last_error[error_type] = now
    
    async def run(self):
        self.running = True
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(1)
        except Exception as e:
            print(f"\n❌ Fatal Error | {e}")
            await self.notifier.error_notification(str(e))
    
    async def stop(self):
        self.running = False
        if self.position:
            await self.close_position("Bot Stop")