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
        print("🆕 Structure break monitoring enabled")
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
        
        # Structure break settings
        self.enable_position_flipping = True
        
        # Break & Retest enhancement
        self.retest_position_multiplier = 1.5  # 1.5x position size for retest patterns
        self.last_retest_signal = None
        
        # Error tracking
        self._last_market_data_error = None
        self._last_position_error = None
        self._last_cycle_error = None
    
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
                
                # Test market data
                test_data = self.get_market_data()
                if test_data is not None:
                    print(f"✅ Market data connection active")
                else:
                    print(f"⚠️ Market data connection issues")
                
                # Test symbol info
                symbol_info = self.get_symbol_info()
                if symbol_info:
                    print(f"✅ Symbol info loaded for {self.symbol}")
                else:
                    print(f"⚠️ Symbol info loading issues")
                
                # Test balance
                balance = self.get_wallet_balance()
                print(f"✅ Wallet balance: ${balance:,.2f}")
                
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
            now = datetime.now()
            if (self._last_market_data_error is None or 
                (now - self._last_market_data_error).seconds > 30):
                print(f"\n❌ API Error | Market Data Failed | {e}")
                self._last_market_data_error = now
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
        """Check current position"""
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
            
            # Check if this is a new position
            if not self.position:
                self.position_start_time = datetime.now()
            
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
            now = datetime.now()
            if (self._last_position_error is None or 
                (now - self._last_position_error).seconds > 30):
                print(f"\n❌ API Error | Position Check Failed | {e}")
                self._last_position_error = now
            self._clear_position()
            return None
    
    def _clear_position(self):
        """Clear position state"""
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
        """Check for structure breaks during active positions"""
        if not self.position or not self.entry_price:
            return None
        
        break_signal = self.strategy.detect_structure_break(
            df, self.position_side, self.entry_price, current_price)
        
        if break_signal:
            break_type = break_signal['break_type']
            break_level = break_signal['break_level']
            
            print(f"\n🚨 STRUCTURE BREAK | {break_type.upper()} | Level: ${break_level:.2f} | Price: ${current_price:.2f}")
            
            # Close current position immediately
            close_success = await self.close_position(f"Structure Break ({break_type})")
            
            if close_success and self.enable_position_flipping:
                # Optional: Flip to opposite position
                flip_action = break_signal['flip_signal']
                
                # Wait a moment for position to clear
                await asyncio.sleep(1)
                
                # Create flip signal
                flip_signal = {
                    'action': flip_action,
                    'price': current_price,
                    'rsi': 50.0,
                    'mfi': 50.0,
                    'timestamp': df.index[-1] if len(df) > 0 else datetime.now(),
                    'structure_stop': None,
                    'signal_type': 'STRUCTURE_FLIP'
                }
                
                print(f"🔄 FLIPPING | {flip_action} after structure break")
                await self.open_position(flip_signal)
            
            return break_signal
        
        return None

    async def handle_risk_management(self, current_price):
        """Handle profit lock activation"""
        if not self.position or not self.entry_price:
            return
            
        # Check for profit lock activation
        if not self.profit_lock_active:
            if self.risk_manager.should_activate_profit_lock(
                self.entry_price, current_price, self.position_side):
                
                self.profit_lock_active = True
                
                # Calculate protected profit
                if self.position_side == 'buy':
                    profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                else:
                    profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                
                protected_value = self.position['size'] * current_price * (profit_pct / 100)
                
                print(f"\n🔒 Profit Lock | +{profit_pct:.1f}% | Trailing: {self.risk_manager.trailing_stop_pct*100:.1f}% | Protected: ${protected_value:.2f}")
                
                # Set trailing stop
                await self._set_trailing_stop(current_price)
                
                await self.notifier.profit_lock_activated(
                    self.symbol, profit_pct, self.risk_manager.trailing_stop_pct * 100
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
            
            if resp.get('retCode') != 0:
                print(f"\n⚠️ SL/TP Warning | Trailing Failed | {resp.get('retMsg')}")
                
        except Exception as e:
            print(f"\n⚠️ SL/TP Warning | Trailing Error | {e}")

    def calculate_enhanced_position_size(self, signal, wallet_balance, entry_price, structure_stop):
        """Calculate position size with Break & Retest enhancement"""
        # Base position size calculation
        base_position_size = self.risk_manager.calculate_position_size(
            wallet_balance, entry_price, structure_stop)
        
        # Apply Break & Retest multiplier for higher probability trades
        if signal.get('signal_type') == 'BREAK_RETEST':
            retest_strength = signal.get('retest_strength', 0.6)
            
            # Higher strength = higher multiplier (1.2x to 1.8x)
            strength_multiplier = 1.0 + (retest_strength * 0.8)  # 1.0 to 1.8x
            enhanced_size = base_position_size * strength_multiplier
            
            # Safety cap: never more than 2x base size
            enhanced_size = min(enhanced_size, base_position_size * 2.0)
            
            print(f"🎯 ENHANCED SIZE | Retest Strength: {retest_strength:.1%} | Multiplier: {strength_multiplier:.1f}x")
            return enhanced_size
        
        return base_position_size

    async def open_position(self, signal):
        """Enhanced position opening with Break & Retest support"""
        try:
            # Close existing position first
            if self.position:
                print(f"\n🔄 Force Close | Clearing position for new signal")
                await self.close_position("Force Close")
                await asyncio.sleep(2)
                
                self.check_position()
                if self.position:
                    print(f"❌ Force Close Failed | Manual intervention required")
                    return False
            
            wallet_balance = self.get_wallet_balance()
            current_price = signal['price']
            structure_stop = signal.get('structure_stop')
            
            # For structure flip signals, calculate structure stop
            if signal.get('signal_type') == 'STRUCTURE_FLIP':
                df = self.get_market_data()
                if df is not None:
                    structure_stop = self.strategy.get_structure_stop(df, signal['action'], current_price)
            
            # Enhanced position sizing for Break & Retest patterns
            position_size = self.calculate_enhanced_position_size(
                signal, wallet_balance, current_price, structure_stop)
            
            info = self.get_symbol_info()
            if not info:
                return False
            
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            # Calculate risk values for display
            side_type = 'long' if signal['action'] == 'BUY' else 'short'
            sl_price = self.risk_manager.get_stop_loss_price(
                current_price, side_type, structure_stop)
            tp_price = self.risk_manager.get_take_profit_price(
                current_price, side_type, structure_stop)
            
            # Calculate actual risk amounts
            actual_risk = float(qty) * abs(current_price - sl_price)
            actual_reward = float(qty) * abs(tp_price - current_price)
            actual_rr = actual_reward / actual_risk if actual_risk > 0 else 4.0
            
            # Enhanced display for Break & Retest patterns
            signal_type_display = signal.get('signal_type', 'RSI_MFI')
            if signal_type_display == 'BREAK_RETEST':
                pattern_type = signal.get('pattern_type', 'retest')
                retest_strength = signal.get('retest_strength', 0.6)
                break_level = signal.get('break_level', current_price)
                
                print(f"\n🎯 BREAK & RETEST ENTRY | {pattern_type.upper()} | Strength: {retest_strength:.1%} | Level: ${break_level:.2f}")
            
            # Store for display
            self.pending_order = {
                'action': signal['action'],
                'qty': qty,
                'price': current_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'risk_amount': actual_risk,
                'reward_amount': actual_reward,
                'rr_ratio': actual_rr,
                'structure_based': structure_stop is not None,
                'structure_stop': structure_stop,
                'signal_type': signal_type_display,
                'enhanced_size': signal.get('signal_type') == 'BREAK_RETEST'
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
                print(f"\n❌ Order Failed | {order.get('retMsg')} | Retry in 5s")
                self.pending_order = None
                return False
            
            # Set stop loss and take profit
            await self._set_stop_and_tp(signal, current_price, info, structure_stop)
            
            # Set initial position state
            self.profit_lock_active = False
            self.entry_price = current_price
            self.position_side = side.lower()
            self.position_start_time = datetime.now()
            
            # Store retest signal for reference
            if signal.get('signal_type') == 'BREAK_RETEST':
                self.last_retest_signal = signal
            
            # Wait for position to be detected
            await asyncio.sleep(1)
            self.check_position()
            
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"\n❌ Order Failed | {e} | Retry in 5s")
            self.pending_order = None
            return False

    async def _set_stop_and_tp(self, signal, current_price, info, structure_stop=None):
        """Set stop loss and take profit"""
        try:
            side = 'long' if signal['action'] == 'BUY' else 'short'
            
            sl_price = self.risk_manager.get_stop_loss_price(
                current_price, side, structure_stop)
            tp_price = self.risk_manager.get_take_profit_price(
                current_price, side, structure_stop)
            
            # For Break & Retest patterns, extend take profit targets
            if signal.get('signal_type') == 'BREAK_RETEST':
                # 1.5x normal target for confirmed retests
                risk_distance = abs(current_price - sl_price)
                extended_reward = risk_distance * 6  # 6:1 instead of 4:1
                
                if side == 'long':
                    tp_price = current_price + extended_reward
                else:
                    tp_price = current_price - extended_reward
                
                print(f"🎯 Extended Target | 6:1 R:R for Break & Retest | TP: ${tp_price:.2f}")
            
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
            
            if stop_resp.get('retCode') != 0:
                print(f"\n⚠️ SL/TP Warning | Price Invalid | Manual monitoring required")
            
        except Exception as e:
            print(f"\n⚠️ SL/TP Warning | Error: {e} | Manual monitoring required")
    
    async def close_position(self, reason="Signal"):
        """Close position"""
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
                print(f"\n❌ Close Failed | {order.get('retMsg')} | Manual intervention required")
                return False
            
            # Calculate duration and result
            duration = "00:00:00"
            if self.position_start_time:
                time_diff = datetime.now() - self.position_start_time
                hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            pnl = self.position.get('unrealized_pnl', 0)
            result = "Win" if pnl > 0 else "Loss"
            
            # Enhanced close message for Break & Retest trades
            close_msg = f"\n📉 CLOSED | {reason} | ⏱️ Duration: {duration} | PnL: {pnl:+.2f} | {result}"
            if self.last_retest_signal:
                retest_strength = self.last_retest_signal.get('retest_strength', 0.6)
                close_msg += f" | Retest: {retest_strength:.1%}"
            
            print(close_msg)
            
            await self.notifier.trade_closed(self.symbol, 0, pnl, reason)
            
            # Clear all position state
            self._clear_position()
            self.last_retest_signal = None
            
            return True
            
        except Exception as e:
            print(f"\n❌ Close Failed | {e} | Manual intervention required")
            return False

    async def handle_signal(self, signal):
        """Enhanced signal handling with Break & Retest priority"""
        if not signal:
            return
        
        # Break & Retest signals have highest priority
        if signal.get('signal_type') == 'BREAK_RETEST':
            print(f"\n🎯 HIGH PRIORITY | Break & Retest Pattern Detected")
            
            # Close any existing position for high-probability retest
            if self.position:
                await self.close_position("Retest Priority")
                await asyncio.sleep(1)
            
            await self.open_position(signal)
            return
        
        # Existing signal logic for RSI/MFI and structure flips
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
            await self.open_position(signal)

    async def run_cycle(self):
        try:
            # Get data
            df = self.get_market_data()
            if df is None or df.empty:
                return
            
            # Check position
            self.check_position()
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Check for structure breaks during active positions
            if self.position:
                structure_break = await self.check_structure_breaks(df, current_price)
                if structure_break:
                    self._display_status(df, current_price)
                    return
            
            # Get signal (now includes Break & Retest detection)
            signal = self.strategy.generate_signal(df)
            
            # Risk management
            if self.position:
                await self.handle_risk_management(current_price)
                self.check_position()
            
            # Display status
            self._display_status(df, current_price)
            
            # Handle signals with priority system
            await self.handle_signal(signal)
                
        except Exception as e:
            now = datetime.now()
            if (self._last_cycle_error is None or 
                (now - self._last_cycle_error).seconds > 30):
                print(f"\n❌ API Error | Connection Lost | Reconnecting... | {e}")
                self._last_cycle_error = now

    def _display_status(self, df, current_price):
        """Enhanced status display with Break & Retest info"""
        # Get indicators
        df_with_indicators = self.strategy.calculate_indicators(df)
        current_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50.0
        current_mfi = df_with_indicators['mfi'].iloc[-1] if 'mfi' in df_with_indicators.columns else 50.0
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        symbol_short = self.symbol.replace('/', '')
        
        # Add retest monitoring status
        retest_status = ""
        if self.strategy.retest_monitoring:
            break_type = self.strategy.retest_monitoring['break_record']['type']
            break_level = self.strategy.retest_monitoring['break_record']['level']
            retest_status = f" | 🔍 Monitoring {break_type} @ ${break_level:.2f}"
        
        if self.pending_order:
            # Enhanced opening info for Break & Retest
            direction_emoji = "📈" if self.pending_order['action'] == 'BUY' else "📉"
            signal_type = self.pending_order.get('signal_type', 'RSI_MFI')
            
            if signal_type == 'BREAK_RETEST':
                signal_emoji = "🎯"
                pattern_info = " (RETEST)"
            elif signal_type == 'STRUCTURE_FLIP':
                signal_emoji = "⚡"
                pattern_info = " (FLIP)"
            else:
                signal_emoji = "🎯"
                pattern_info = ""
            
            # Line 1: Opening info
            opening_line = f"[{timestamp}] {symbol_short} | RSI: {current_rsi:.1f} | MFI: {current_mfi:.1f} | {direction_emoji} {signal_emoji} {self.pending_order['action']}{pattern_info} {self.pending_order['qty']} @ ${self.pending_order['price']:.2f}"
            print(f"\n{opening_line}")
            
            # Line 2: Risk details
            stop_type = "Structure" if self.pending_order.get('structure_based') else "Fixed"
            stop_distance_pct = abs(self.pending_order['price'] - self.pending_order['sl_price']) / self.pending_order['price'] * 100
            
            size_info = ""
            if self.pending_order.get('enhanced_size'):
                size_info = " (Enhanced) "
            
            risk_line = (f"💰 Risk: ${self.pending_order['risk_amount']:.0f}{size_info}| "
                        f"SL: ${self.pending_order['sl_price']:.2f} ({stop_type} {stop_distance_pct:.1f}%) | "
                        f"TP: ${self.pending_order['tp_price']:.2f} (+${self.pending_order['reward_amount']:.0f}) | "
                        f"R:R 1:{self.pending_order['rr_ratio']:.1f}")
            print(f"{risk_line}")
            
            # Clear pending order
            self.pending_order = None
            
        elif self.position:
            # Position monitoring
            if self.position_start_time:
                duration = datetime.now() - self.position_start_time
                hours, remainder = divmod(int(duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = "00:00:00"
            
            pnl = self.position['unrealized_pnl']
            lock_status = ' 🔒' if self.profit_lock_active else ''
            
            # Add retest info if applicable
            retest_info = ""
            if self.last_retest_signal:
                strength = self.last_retest_signal.get('retest_strength', 0.6)
                retest_info = f" | 🎯 {strength:.1%}"
            
            monitor_line = f"⏱️ {duration_str} | ${current_price:.2f} | PnL: {pnl:+.2f}{lock_status}{retest_info}"
            print(f"\r{monitor_line}", end='', flush=True)
            
        else:
            # No position - market status with retest monitoring
            status = f"[{timestamp}] {symbol_short} | RSI: {current_rsi:.1f} | MFI: {current_mfi:.1f} | No Position{retest_status}"
            print(f"\r{status}", end='', flush=True)
    
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