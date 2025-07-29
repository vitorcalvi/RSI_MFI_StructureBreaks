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
        self.current_atr_pct = 0
        
        # Ranging market tracking
        self.ranging_cycles = 0
        self.last_trend = None
        
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
                    
                    balance = self.get_account_balance()
                    pnl_pct = self.risk_manager.calculate_account_pnl_pct(unrealized_pnl, balance)
                    
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
                self.ranging_cycles = 0
                self.last_trend = None
            self.position = None
            self.profit_lock_active = False
            return None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            return None
    
    def update_ranging_tracking(self, current_trend):
        """Update ranging market cycle tracking"""
        if self.last_trend == current_trend:
            if current_trend == "SIDEWAYS":
                self.ranging_cycles += 1
            else:
                self.ranging_cycles = 0
        else:
            self.ranging_cycles = 0
        
        self.last_trend = current_trend
    
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
        
        if self.risk_manager.should_switch_position(pnl_pct):
            current_side = self.position['side']
            new_side = 'SELL' if current_side == 'Buy' else 'BUY'
            
            print(f"\n⚠️ Loss threshold reached: {pnl_pct:.2f}% (limit: {self.risk_manager.loss_switch_threshold}%)")
            print(f"🔄 Switching to {new_side}")
            
            await self.close_position("Loss Limit")
            await asyncio.sleep(2)
            
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

    async def check_smart_exit_conditions(self, current_rsi, current_mfi, current_trend):
        """FIXED: Check all smart exit conditions in one place"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        position_side = self.position['side']
        
        # 1. Check ranging market exit
        should_exit_ranging, ranging_reason = self.risk_manager.should_exit_ranging_market(
            pnl_pct, current_trend, current_rsi, current_mfi, 
            position_side, self.ranging_cycles
        )
        
        if should_exit_ranging:
            print(f"\n🎯 SMART EXIT: {ranging_reason}")
            await self.close_position("Smart Exit")
            return
        
        # 2. Check profit protection (high profit)
        if self.risk_manager.should_take_profit_protection(pnl_pct):
            print(f"\n💰 PROFIT PROTECTION: {pnl_pct:.2f}% (threshold: {self.risk_manager.profit_protection_threshold}%)")
            await self.close_position("Profit Protection")
            return

    async def check_profit_lock(self, current_price):
        """Check if profit lock should be activated"""
        if not self.position:
            return
            
        pnl_pct = self.position['unrealized_pnl_pct']
        
        # Dynamic threshold calculation
        if self.current_atr_pct > 0:
            dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
            if pnl_pct > 0 and not self.profit_lock_active:
                print(f" [Lock: {pnl_pct:.1f}%/{dynamic_threshold:.1f}%]", end='')
        else:
            if pnl_pct > 0 and not self.profit_lock_active:
                print(f" [Lock: {pnl_pct:.1f}%/{self.risk_manager.profit_lock_threshold:.1f}%]", end='')
        
        # Check profit lock activation
        if not self.profit_lock_active and self.risk_manager.should_activate_profit_lock(pnl_pct, self.current_atr_pct):
            self.profit_lock_active = True
            
            if self.current_atr_pct > 0:
                dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
                print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}% (ATR-dynamic: {dynamic_threshold:.1f}%)")
            else:
                print(f"\n🔓 PROFIT LOCK ACTIVATED! P&L: {pnl_pct:.2f}% (static: {self.risk_manager.profit_lock_threshold:.1f}%)")
            
            await self.notifier.profit_lock_activated(
                self.symbol, pnl_pct, self.risk_manager.trailing_stop_distance * 100
            )
            
            # Set trailing stop
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
            
            # Set initial stops
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
            self.ranging_cycles = 0
            self.last_trend = None
            
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
            
            if reason == "Profit Protection" and self.risk_manager.should_take_profit_protection(pnl_pct):
                self.reversal_cooldown_cycles = self.risk_manager.reversal_cooldown_cycles
            
            await self.notifier.trade_closed(self.symbol, pnl_pct, pnl, reason)
            
            self.position = None
            self.profit_lock_active = False
            self.ranging_cycles = 0
            self.last_trend = None
            return True
            
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False

    async def run_cycle(self):
        """Run trading cycle with FIXED smart exit logic"""
        try:
            df = self.get_market_data()
            if df is None or df.empty:
                print("⚠️ No market data available")
                return
            
            self.check_position()
            signal = self.strategy.generate_signal(df)
            
            try:
                current_price = df['close'].iloc[-1]
            except (IndexError, KeyError):
                print("⚠️ Cannot get current price")
                return
            
            # Get ATR from strategy
            try:
                if hasattr(self.strategy, 'current_atr_pct'):
                    self.current_atr_pct = self.strategy.current_atr_pct
                else:
                    self.current_atr_pct = 0
            except:
                self.current_atr_pct = 0
            
            # Get indicators
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
            
            # Update ranging tracking
            self.update_ranging_tracking(current_trend)
            
            # Handle cooldown
            if self.reversal_cooldown_cycles > 0:
                self.reversal_cooldown_cycles -= 1
                if self.reversal_cooldown_cycles == 0:
                    print(f"\n🔓 Cooldown ended")
            
            # FIXED: Check all exit conditions if we have a position
            if self.position:
                # 1. Check smart exit conditions FIRST
                await self.check_smart_exit_conditions(current_rsi, current_mfi, current_trend)
                
                # 2. Only check other conditions if position still exists
                if self.position:
                    await self.check_profit_lock(current_price)
                    await self.check_loss_switch()
            
            # Display status
            position_info = 'None'
            cooldown_info = ''
            ranging_info = ''
            trend_emoji = {"UP": "🟢", "DOWN": "🔴", "SIDEWAYS": "🟡", "UNKNOWN": "⚪"}.get(current_trend, "⚪")
            
            if self.reversal_cooldown_cycles > 0:
                cooldown_info = f' [Cooldown: {self.reversal_cooldown_cycles}]'
            
            if self.position:
                pnl_pct = self.position['unrealized_pnl_pct']
                lock_status = '🔒' if self.profit_lock_active else ''
                position_info = f"{self.position['side']} ({pnl_pct:+.2f}%) {lock_status}"
            
            if self.ranging_cycles > 0:
                ranging_info = f' [Ranging: {self.ranging_cycles}]'
            
            # ATR and Lock threshold display
            if self.current_atr_pct > 0:
                atr_display = f"ATR:{self.current_atr_pct:.1f}%"
                dynamic_threshold = self.risk_manager.get_dynamic_profit_lock_threshold(self.current_atr_pct)
                lock_display = f"Lock@{dynamic_threshold:.1f}%"
            else:
                atr_display = "ATR:N/A"
                lock_display = f"Lock@{self.risk_manager.profit_lock_threshold:.1f}%"
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                f"{trend_emoji}{current_trend} | {atr_display} | {lock_display} | "
                f"{position_info}{cooldown_info}{ranging_info}", 
                end='', flush=True)
            
            # Handle signals - SIMPLIFIED LOGIC
            if signal:
                 await self.handle_signal_zora_optimized(signal)
                
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
    
    async def handle_signal_simple(self, signal):
        """FIXED: Simplified signal handling - more responsive"""
        signal_action = signal['action']
        
        if self.position:
            current_side = self.position['side']
            pnl_pct = self.position['unrealized_pnl_pct']
            
            # Check if signal is opposite to current position
            should_reverse = (
                (current_side == 'Buy' and signal_action == 'SELL') or 
                (current_side == 'Sell' and signal_action == 'BUY')
            )
            
            if should_reverse:
                # SIMPLIFIED: More aggressive reversal logic
                if pnl_pct >= 0.1:  # Any profit > 0.1% = take it
                    print(f"\n💰 Taking {pnl_pct:.2f}% profit on opposite signal")
                    await self.close_position("Profit Signal")
                    await asyncio.sleep(2)
                    await self.open_position(signal)
                elif pnl_pct <= -2.0:  # Loss > 2% = reverse to limit damage
                    print(f"\n🔄 Reversing losing position: {pnl_pct:.2f}%")
                    await self.close_position("Loss Reversal")
                    await asyncio.sleep(2)
                    await self.open_position(signal)
                else:
                    print(f"\n⏸️ Signal ignored - P&L: {pnl_pct:.2f}% (between -2% and +0.1%)")
        else:
            # No position - open new one if not in cooldown
            if self.reversal_cooldown_cycles > 0:
                print(f"\n⏸️ Cooldown active ({self.reversal_cooldown_cycles} cycles)")
            else:
                atr_info = f" (ATR: {self.current_atr_pct:.1f}%)" if self.current_atr_pct > 0 else ""
                print(f"\n🎯 Signal: {signal_action} @ ${signal['price']:.4f}{atr_info}")
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
        

# Add this method to your TradeEngine class:

    async def handle_signal_zora_optimized(self, signal):
        """ZORA-optimized signal handling with volume and MACD confirmation"""
        signal_action = signal['action']
        
        # Get current market data for validation
        df = self.get_market_data()
        if df is None or len(df) < 10:
            return
        
        # Calculate volume ratio (need last 10 periods for average)
        recent_volumes = df['volume'].tail(10).tolist()
        current_volume = df['volume'].iloc[-1]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Get MACD data (if available from strategy)
        macd = macd_signal = 0
        try:
            df_with_indicators = self.strategy.calculate_indicators(df)
            if df_with_indicators is not None and 'macd' in df_with_indicators.columns:
                macd = df_with_indicators['macd'].iloc[-1]
                macd_signal = df_with_indicators['macd_signal'].iloc[-1] if 'macd_signal' in df_with_indicators.columns else 0
        except:
            pass
        
        # Get current RSI, MFI, trend
        current_rsi = current_mfi = 50.0
        current_trend = "UNKNOWN"
        try:
            df_with_indicators = self.strategy.calculate_indicators(df)
            if df_with_indicators is not None:
                current_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50.0
                current_mfi = df_with_indicators['mfi'].iloc[-1] if 'mfi' in df_with_indicators.columns else 50.0
                current_trend = df_with_indicators['trend'].iloc[-1] if 'trend' in df_with_indicators.columns else "UNKNOWN"
        except:
            pass
        
        # Validate signal using ZORA-specific criteria
        if hasattr(self.risk_manager, 'is_valid_zora_signal'):
            is_valid, reason = self.risk_manager.is_valid_zora_signal(
                current_rsi, current_mfi, current_trend, volume_ratio, macd, macd_signal
            )
            
            if not is_valid:
                print(f"\n⏸️ ZORA Signal rejected: {reason}")
                return
        
        if self.position:
            current_side = self.position['side']
            pnl_pct = self.position['unrealized_pnl_pct']
            
            # Check if signal is opposite to current position
            should_reverse = (
                (current_side == 'Buy' and signal_action == 'SELL') or 
                (current_side == 'Sell' and signal_action == 'BUY')
            )
            
            if should_reverse:
                # ZORA-optimized reversal logic
                if pnl_pct >= 0.05:  # Any profit > 0.05% = take it (more conservative)
                    print(f"\n💰 ZORA: Taking {pnl_pct:.2f}% profit on opposite signal")
                    await self.close_position("ZORA Profit Signal")
                    await asyncio.sleep(2)
                    await self.open_position(signal)
                elif pnl_pct <= -1.5:  # Loss > 1.5% = reverse to limit damage (tighter)
                    print(f"\n🔄 ZORA: Reversing losing position: {pnl_pct:.2f}%")
                    await self.close_position("ZORA Loss Reversal")
                    await asyncio.sleep(2)
                    await self.open_position(signal)
                else:
                    print(f"\n⏸️ ZORA Signal ignored - P&L: {pnl_pct:.2f}% (between -1.5% and +0.05%)")
        else:
            # No position - open new one if not in cooldown
            if self.reversal_cooldown_cycles > 0:
                print(f"\n⏸️ Cooldown active ({self.reversal_cooldown_cycles} cycles)")
            else:
                atr_info = f" (ATR: {self.current_atr_pct:.1f}%)" if self.current_atr_pct > 0 else ""
                vol_info = f" (Vol: {volume_ratio:.1f}x)" if volume_ratio > 1.0 else ""
                print(f"\n🎯 ZORA Signal: {signal_action} @ ${signal['price']:.4f}{atr_info}{vol_info}")
                await self.open_position(signal)

    # Replace your current handle_signal_simple method call with:
    # await self.handle_signal_zora_optimized(signal)