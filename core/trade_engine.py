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
        
        # Configuration (use centralized symbol)
        self.symbol = self.strategy.symbol  # From JSON config
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
        self.current_atr_pct = 0
        
    def connect(self):
        """Initialize exchange connection"""
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
            resp = self.exchange.set_leverage(
                category="linear",
                symbol=self.linear,
                buyLeverage=str(self.risk_manager.leverage),
                sellLeverage=str(self.risk_manager.leverage)
            )
            
            ret_code = resp.get('retCode', 0)
            if ret_code in [0, 110043, 110036]:
                pass  # Leverage set successfully or already set
            else:
                print(f"⚠️ Leverage: {resp.get('retMsg')}")
        except Exception as e:
            print(f"⚠️ Leverage error: {str(e)[:30]}...")
    
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
            
            if not klines.get('result') or not klines['result'].get('list'):
                print("❌ No market data in response")
                return None
                
            data = klines['result']['list']
            if not data:
                print("❌ Empty market data")
                return None
            
            return self._process_market_data(data)
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return None
    
    def _process_market_data(self, data):
        """Process raw market data into DataFrame"""
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df = df.set_index('timestamp')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df = df.sort_index()
        
        if df.empty:
            print("❌ Empty DataFrame after processing")
            return None
            
        return df
    
    def get_current_price(self, df):
        """Extract current price from market data"""
        try:
            return df['close'].iloc[-1]
        except (IndexError, KeyError):
            print("⚠️ Cannot get current price")
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
                    return self._build_position_data(position)
            
            # No position
            if self.position is not None:
                print(f"\n📉 Position closed")
            self._clear_position_state()
            return None
            
        except Exception as e:
            print(f"❌ Position check error: {e}")
            return None
    
    def _build_position_data(self, position):
        """Build position data structure"""
        avg_price = float(position['avgPrice'])
        unrealized_pnl = float(position['unrealisedPnl'])
        
        balance = self.get_account_balance()
        pnl_pct = self.risk_manager.calculate_account_pnl_pct(unrealized_pnl, balance)
        
        self.position = {
            'side': position['side'],
            'size': float(position['size']),
            'avg_price': avg_price,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': pnl_pct
        }
        return self.position
    
    def _clear_position_state(self):
        """Clear position-related state"""
        self.position = None
        self.profit_lock_active = False
    
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

    async def check_profit_protection(self):
        """Check profit protection only"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        
        if self.risk_manager.should_take_profit_protection(pnl_pct):
            print(f"\n💰 PROFIT PROTECTION: {pnl_pct:.2f}% (threshold: {self.risk_manager.profit_protection_threshold}%)")
            await self.close_position("Profit Protection")

    async def check_profit_lock(self, current_price):
        """Check if profit lock should be activated"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        
        # Display current status
        self._display_profit_lock_status(pnl_pct)
        
        # Check activation
        if not self.profit_lock_active and self.risk_manager.should_activate_profit_lock(pnl_pct, self.current_atr_pct):
            await self._activate_profit_lock(pnl_pct, current_price)
    
    def _display_profit_lock_status(self, pnl_pct):
        """Display profit lock status"""
        if pnl_pct > 0 and not self.profit_lock_active:
            if self.current_atr_pct > 0:
                dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
                print(f" [Lock: {pnl_pct:.1f}%/{dynamic_threshold:.1f}%]", end='')
            else:
                print(f" [Lock: {pnl_pct:.1f}%/{self.risk_manager.profit_lock_threshold:.1f}%]", end='')
    
    async def _activate_profit_lock(self, pnl_pct, current_price):
        """Activate profit lock with trailing stop"""
        self.profit_lock_active = True
        
        if self.current_atr_pct > 0:
            dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
            print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}% (ATR-dynamic: {dynamic_threshold:.1f}%)")
        else:
            print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}% (static: {self.risk_manager.profit_lock_threshold:.1f}%)")
        
        await self.notifier.profit_lock_activated(
            self.symbol, pnl_pct, self.risk_manager.trailing_stop_distance * 100
        )
        
        await self._set_trailing_stop(current_price)
        self.last_trailing_update = datetime.now()
    
    async def _set_trailing_stop(self, current_price):
        """Set trailing stop"""
        info = self.get_symbol_info()
        if info:
            trailing_distance = self.risk_manager.get_trailing_stop_distance_absolute(current_price)
            formatted_trailing_distance = self.format_price(info, trailing_distance)
            
            resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                takeProfit="",
                stopLoss="",
                trailingStop=formatted_trailing_distance,
                activePrice=self.format_price(info, current_price)
            )
            
            if resp.get('retCode') == 0:
                print(f"✅ Trailing stop set: {formatted_trailing_distance} distance")
            else:
                print(f"❌ Trailing stop failed: {resp.get('retMsg')}")

    async def open_position(self, signal):
        """Open position"""
        try:
            # Get market data
            balance = self.get_account_balance()
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate position
            position_size = self.risk_manager.calculate_position_size(balance, current_price)
            info = self.get_symbol_info()
            if not info:
                return False
            
            # Execute order
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            print(f"\n📈 Opening {side}: {qty} @ ${current_price:.4f}")
            
            success = await self._execute_market_order(side, qty)
            if not success:
                return False
            
            # Set ONLY stop loss - NO TAKE PROFIT
            await self._set_initial_stops(signal, current_price, info)
            
            # Notify and update state
            self.profit_lock_active = False
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def _execute_market_order(self, side, qty):
        """Execute market order"""
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
        return True
    
    async def _set_initial_stops(self, signal, current_price, info):
        """Set ONLY stop loss - NO TAKE PROFIT"""
        try:
            # Calculate ONLY stop loss
            if signal['action'] == 'BUY':
                sl = self.risk_manager.get_stop_loss(current_price, 'long')
            else:
                sl = self.risk_manager.get_stop_loss(current_price, 'short')
            
            # Set ONLY stop loss - NO TP
            stop_resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                stopLoss=self.format_price(info, sl),
                slTriggerBy="LastPrice"
                # NO takeProfit parameter - REMOVED COMPLETELY
            )
            
            if stop_resp.get('retCode') == 0:
                print(f"✅ Stop Loss set: ${sl:.6f} (NO TP - Hold until signal/protection)")
            else:
                print(f"⚠️ Stop setting failed: {stop_resp.get('retMsg')} (continuing)")
                
        except Exception as e:
            print(f"⚠️ Stop setting error: {e} (continuing)")
    
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
            
            # Handle post-close actions
            await self._handle_position_close(reason)
            return True
            
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False
    
    async def _handle_position_close(self, reason):
        """Handle post-position-close actions - UNIFIED cooldown"""
        pnl = self.position.get('unrealized_pnl', 0)
        pnl_pct = self.position.get('unrealized_pnl_pct', 0)
        
        if reason == "Profit Protection" and self.risk_manager.should_take_profit_protection(pnl_pct):
            self.reversal_cooldown_cycles = self.risk_manager.reversal_cooldown_cycles
        
        await self.notifier.trade_closed(self.symbol, pnl_pct, pnl, reason)
        self._clear_position_state()
    
    def extract_indicators(self, df):
        """Extract current indicators from market data"""
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
        
        return current_rsi, current_mfi, current_trend
    
    def update_atr_data(self):
        """Update ATR data from strategy"""
        try:
            if hasattr(self.strategy, 'current_atr_pct'):
                self.current_atr_pct = self.strategy.current_atr_pct
            else:
                self.current_atr_pct = 0
        except:
            self.current_atr_pct = 0
    
    def handle_cooldown(self):
        """Handle reversal cooldown logic"""
        if self.reversal_cooldown_cycles > 0:
            self.reversal_cooldown_cycles -= 1
            if self.reversal_cooldown_cycles == 0:
                print(f"\n🔓 Cooldown ended")

    async def handle_signal(self, signal):
        """Handle trading signal"""
        if not signal:
            return
            
        # Validate signal
        if not await self._validate_signal(signal):
            return
        
        # Process signal based on current position
        if self.position:
            await self._handle_signal_with_position(signal)
        else:
            await self._handle_signal_no_position(signal)
    
    async def _validate_signal(self, signal):
        """Validate signal with ZORA-specific criteria"""
        df = self.get_market_data()
        if df is None or len(df) < 10:
            return False
        
        # Get validation data
        volume_ratio = self._calculate_volume_ratio(df)
        macd, macd_signal = self._get_macd_data(df)
        current_rsi, current_mfi, current_trend = self.extract_indicators(df)
        
        # Validate
        if hasattr(self.risk_manager, 'is_valid_zora_signal'):
            is_valid, reason = self.risk_manager.is_valid_zora_signal(
                current_rsi, current_mfi, current_trend, volume_ratio, macd, macd_signal
            )
            
            if not is_valid:
                print(f"\n⏸️ ZORA Signal rejected: {reason}")
                return False
        
        return True
    
    def _calculate_volume_ratio(self, df):
        """Calculate volume ratio"""
        recent_volumes = df['volume'].tail(10).tolist()
        current_volume = df['volume'].iloc[-1]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _get_macd_data(self, df):
        """Get MACD data"""
        macd = macd_signal = 0
        try:
            df_with_indicators = self.strategy.calculate_indicators(df)
            if df_with_indicators is not None and 'macd' in df_with_indicators.columns:
                macd = df_with_indicators['macd'].iloc[-1]
                macd_signal = df_with_indicators['macd_signal'].iloc[-1] if 'macd_signal' in df_with_indicators.columns else 0
        except:
            pass
        return macd, macd_signal
    
    async def _handle_signal_with_position(self, signal):
        """Handle signal when we have a position"""
        current_side = self.position['side']
        pnl_pct = self.position['unrealized_pnl_pct']
        signal_action = signal['action']
        
        # Check if signal is opposite to current position
        should_reverse = (
            (current_side == 'Buy' and signal_action == 'SELL') or 
            (current_side == 'Sell' and signal_action == 'BUY')
        )
        
        if should_reverse:
            await self._handle_position_reversal(signal, pnl_pct)
    
    async def _handle_position_reversal(self, signal, pnl_pct):
        """Handle position reversal logic - UNIFIED with RiskManager"""
        if self.risk_manager.should_reverse_for_profit(pnl_pct):
            print(f"\n💰 ZORA: Taking {pnl_pct:.2f}% profit on opposite signal (threshold: {self.risk_manager.profit_reversal_threshold:.2f}%)")
            await self.close_position("ZORA Profit Signal")
            await asyncio.sleep(2)
            await self.open_position(signal)
        elif self.risk_manager.should_reverse_for_loss(pnl_pct):
            print(f"\n🔄 ZORA: Reversing losing position: {pnl_pct:.2f}% (threshold: {self.risk_manager.loss_reversal_threshold:.2f}%)")
            await self.close_position("ZORA Loss Reversal")
            await asyncio.sleep(2)
            await self.open_position(signal)
        else:
            profit_thresh = self.risk_manager.profit_reversal_threshold
            loss_thresh = self.risk_manager.loss_reversal_threshold
            print(f"\n⏸️ ZORA Signal ignored - P&L: {pnl_pct:.2f}% (between {loss_thresh:.2f}% and +{profit_thresh:.2f}%)")
    
    async def _handle_signal_no_position(self, signal):
        """Handle signal when we have no position"""
        if self.reversal_cooldown_cycles > 0:
            print(f"\n⏸️ Cooldown active ({self.reversal_cooldown_cycles} cycles)")
        else:
            atr_info = f" (ATR: {self.current_atr_pct:.1f}%)" if self.current_atr_pct > 0 else ""
            print(f"\n🎯 ZORA Signal: {signal['action']} @ ${signal['price']:.4f}{atr_info}")
            await self.open_position(signal)

    def create_status_display(self, current_price, current_rsi, current_mfi, current_trend):
        """Create status display string"""
        # Position info
        position_info = 'None'
        cooldown_info = ''
        
        if self.reversal_cooldown_cycles > 0:
            cooldown_info = f' [Cooldown: {self.reversal_cooldown_cycles}]'
        
        if self.position:
            pnl_pct = self.position['unrealized_pnl_pct']
            lock_status = '🔒' if self.profit_lock_active else ''
            position_info = f"{self.position['side']} ({pnl_pct:+.2f}%) {lock_status}"
        else:
            # Show what we're waiting for when no position
            oversold = self.strategy.params['oversold_level']
            overbought = self.strategy.params['overbought_level']
            require_trend = self.strategy.params.get('require_trend', False)
            
            if require_trend and current_trend == 'DOWN' and current_rsi <= oversold + 5:
                position_info = 'Waiting: UP trend'
            elif current_rsi > oversold and current_mfi > oversold:
                position_info = f'Waiting: RSI<{oversold}'
            else:
                position_info = 'None'
        
        # Trend emoji with indicator explanation - FIXED
        trend_indicators = {
            "UP": "🟢 EMA12>EMA26", 
            "DOWN": "🔴 EMA12<EMA26", 
            "SIDEWAYS": "🟡 EMA12≈EMA26", 
            "UNKNOWN": "⚪ Unknown"
        }
        trend_display = trend_indicators.get(current_trend, "⚪Unknown")
        
        # ATR and Lock threshold display
        if self.current_atr_pct > 0:
            atr_display = f"ATR:{self.current_atr_pct:.1f}%"
            dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
            lock_display = f"Lock@{dynamic_threshold:.1f}%"
        else:
            atr_display = "ATR:N/A"
            lock_display = f"Lock@{self.risk_manager.profit_lock_threshold:.1f}%"
        
        return (f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                f"{trend_display} | {atr_display} | {lock_display} | "
                f"{position_info}{cooldown_info}")

    async def run_cycle(self):
        """Main trading cycle - CLEAN and focused"""
        try:
            # 1. Get market data
            df = self.get_market_data()
            if df is None or df.empty:
                print("⚠️ No market data available")
                return
            
            # 2. Update position and extract data
            self.check_position()
            signal = self.strategy.generate_signal(df)
            current_price = self.get_current_price(df)
            if current_price is None:
                return
            
            # 3. Update indicators and ATR
            self.update_atr_data()
            current_rsi, current_mfi, current_trend = self.extract_indicators(df)
            
            # 4. Handle cooldown
            self.handle_cooldown()
            
            # 5. Risk management checks (if we have position)
            if self.position:
                await self.check_profit_protection()
                if self.position:  # Check if still exists after profit protection
                    await self.check_profit_lock(current_price)
            
            # 6. Display status
            status = self.create_status_display(current_price, current_rsi, current_mfi, current_trend)
            print(f"\r{status}", end='', flush=True)
            
            # 7. Handle signals
            await self.handle_signal(signal)
                
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
    
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