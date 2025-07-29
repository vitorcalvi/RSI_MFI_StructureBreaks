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
        self.strategy = RSIMFICloudStrategy()
        self.notifier = TelegramNotifier()
        
        self.symbol = self.strategy.symbol
        self.linear = self.symbol.replace('/', '')
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
        self.reversal_cooldown_cycles = 0
    
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
                self._set_leverage()
                return True
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def _set_leverage(self):
        try:
            self.exchange.set_leverage(
                category="linear",
                symbol=self.linear,
                buyLeverage=str(self.risk_manager.leverage),
                sellLeverage=str(self.risk_manager.leverage)
            )
        except:
            pass
    
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
    
    def get_account_balance(self):
        try:
            resp = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if resp.get('retCode') == 0 and resp.get('result', {}).get('list'):
                return float(resp['result']['list'][0].get('totalEquity', 0))
            return 0
        except:
            return 0

    def get_wallet_balance_only(self):
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
            if pos_resp.get('retCode') == 0 and pos_resp['result']['list']:
                position = pos_resp['result']['list'][0]
                position_size = float(position['size'])
                
                if position_size > 0:
                    # Get values from API
                    unrealized_pnl = float(position['unrealisedPnl'])
                    avg_price = float(position['avgPrice'])
                    side = position['side']
                    
                    # Calculate position metrics
                    position_value = position_size * avg_price
                    margin_used = position_value / self.risk_manager.leverage
                    
                    # FIXED: Calculate ROE (Return on Equity) - leveraged P&L percentage
                    roe_pct = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
                    
                    
                    self.position = {
                        'side': side,
                        'size': position_size,
                        'avg_price': avg_price,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': roe_pct,  # Now ROE percentage
                        'margin_used': margin_used,
                        'position_value': position_value
                    }
                    
                    return self.position
            
            # No position
            self.position = None
            self.profit_lock_active = False
            return None
            
        except Exception as e:
            print(f"❌ Position check error: {e}")
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
        price = round(price / tick) * tick
        
        tick_str = f"{tick:.20f}".rstrip('0').rstrip('.')
        decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{decimals}f}"

    async def handle_risk_management(self, current_price):
        """Risk management with ROE-based thresholds"""
        if not self.position:
            return
            
        roe_pct = self.position['unrealized_pnl_pct']
        
        # PRIORITY 1: Profit Protection (HIGHEST)
        if self.risk_manager.should_take_profit_protection(roe_pct):
            print(f"\n💰 PROFIT PROTECTION: {roe_pct:.2f}% ROE")
            await self.close_position("Profit Protection")
            return
        
        # PRIORITY 2: Profit Lock (only if no protection)
        if not self.profit_lock_active and self.risk_manager.should_activate_profit_lock(roe_pct):
            self.profit_lock_active = True
            print(f"\n🔓 PROFIT LOCK ACTIVATED! ROE: {roe_pct:.2f}%")
            
            await self.notifier.profit_lock_activated(
                self.symbol, roe_pct, self.risk_manager.trailing_stop_distance * 100
            )
            
            await self._set_trailing_stop(current_price)

    async def _set_trailing_stop(self, current_price):
        info = self.get_symbol_info()
        if info:
            trailing_distance = self.risk_manager.get_trailing_stop_distance_absolute(current_price)
            formatted_trailing_distance = self.format_price(info, trailing_distance)
            
            resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                trailingStop=formatted_trailing_distance
            )
            
            if resp.get('retCode') == 0:
                print(f"✅ Trailing stop set: {formatted_trailing_distance}")

    async def open_position(self, signal):
        try:
            wallet_balance = self.get_wallet_balance_only()
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            position_size = self.risk_manager.calculate_position_size(wallet_balance, current_price)
            info = self.get_symbol_info()
            if not info:
                return False
            
            qty = self.format_qty(info, position_size)
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            print(f"\n📈 Opening {side}: {qty} @ ${current_price:.4f}")
            
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
            
            # Wait a moment for position to be created
            await asyncio.sleep(2)
            
            # Set stop loss
            await self._set_stop_loss(signal, current_price, info)
            
            self.profit_lock_active = False
            await self.notifier.trade_opened(self.symbol, current_price, float(qty), side)
            return True
            
        except Exception as e:
            print(f"❌ Position open error: {e}")
            return False
    
    async def _set_stop_loss(self, signal, current_price, info):
        try:
            if signal['action'] == 'BUY':
                sl = self.risk_manager.get_stop_loss(current_price, 'long')
            else:
                sl = self.risk_manager.get_stop_loss(current_price, 'short')
            
            stop_resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.linear,
                positionIdx=0,
                stopLoss=self.format_price(info, sl),
                slTriggerBy="LastPrice"
            )
            
            if stop_resp.get('retCode') == 0:
                print(f"✅ Stop Loss: ${sl:.6f}")
            
        except Exception as e:
            print(f"⚠️ Stop setting error: {e}")
    
    async def close_position(self, reason="Signal"):
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
                return False
            
            # Handle cooldown for loss reversals only
            roe_pct = self.position.get('unrealized_pnl_pct', 0)
            if reason == "Loss Reversal":
                self.reversal_cooldown_cycles = self.risk_manager.reversal_cooldown_cycles
            
            pnl = self.position.get('unrealized_pnl', 0)
            await self.notifier.trade_closed(self.symbol, roe_pct, pnl, reason)
            
            self.position = None
            self.profit_lock_active = False
            return True
            
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False

    async def handle_signal(self, signal):
        if not signal:
            return
        
        if self.position:
            await self._handle_signal_with_position(signal)
        else:
            await self._handle_signal_no_position(signal)
    
    async def _handle_signal_with_position(self, signal):
        """Only reverse on loss, no profit reversals"""
        current_side = self.position['side']
        roe_pct = self.position['unrealized_pnl_pct']
        signal_action = signal['action']
        
        # Check if signal is opposite
        should_reverse = (
            (current_side == 'Buy' and signal_action == 'SELL') or 
            (current_side == 'Sell' and signal_action == 'BUY')
        )
        
        if should_reverse:
            # Only reverse on loss - no profit reversals
            if self.risk_manager.should_reverse_for_loss(roe_pct):
                print(f"\n🔄 Reversing losing position: {roe_pct:.2f}% ROE")
                await self.close_position("Loss Reversal")
                await asyncio.sleep(1)
                await self.open_position(signal)
    
    async def _handle_signal_no_position(self, signal):
        if self.reversal_cooldown_cycles > 0:
            print(f"\n⏸️ Cooldown active ({self.reversal_cooldown_cycles} cycles)")
        else:
            print(f"\n🎯 Signal: {signal['action']} @ ${signal['price']:.4f}")
            await self.open_position(signal)

    async def run_cycle(self):
        try:
            # Get data
            df = self.get_market_data()
            if df is None or df.empty:
                return
            
            # Update position (get fresh data)
            position = self.check_position()
            
            # Get signal and current price
            signal = self.strategy.generate_signal(df)
            current_price = df['close'].iloc[-1]
            
            # Handle cooldown
            if self.reversal_cooldown_cycles > 0:
                self.reversal_cooldown_cycles -= 1
            
            # Risk management checks (using fresh position data)
            if position:
                await self.handle_risk_management(current_price)
            
            # Display status
            self._display_status(df, current_price)
            
            # Handle signals (only if position still exists after risk management)
            if self.position or not position:
                await self.handle_signal(signal)
                
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
    
    def _display_status(self, df, current_price):
        # Get indicators
        df_with_indicators = self.strategy.calculate_indicators(df)
        current_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50.0
        current_mfi = df_with_indicators['mfi'].iloc[-1] if 'mfi' in df_with_indicators.columns else 50.0
        current_trend = df_with_indicators['trend'].iloc[-1] if 'trend' in df_with_indicators.columns else "UNKNOWN"
        
        # Position info with risk zone
        position_info = 'None'
        if self.position:
            roe_pct = self.position['unrealized_pnl_pct']
            risk_zone = self.risk_manager.get_risk_zone(roe_pct)
            zone_emoji = {
                'PROFIT_PROTECTION': '💰',
                'PROFIT_LOCK': '🔒',
                'LOSS_REVERSAL': '🔄',
                'NORMAL': ''
            }.get(risk_zone, '')
            
            position_info = f"{self.position['side']} ({roe_pct:+.2f}%) {zone_emoji}"
        
        # Trend display
        trend_emoji = {"UP": "🟢", "DOWN": "🔴", "SIDEWAYS": "🟡"}.get(current_trend, "⚪")
        
        # Cooldown info
        cooldown_info = f' [Cooldown: {self.reversal_cooldown_cycles}]' if self.reversal_cooldown_cycles > 0 else ''
        
        status = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                  f"{trend_emoji} | {position_info}{cooldown_info}")
        
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
        
        await self.notifier.bot_stopped()
        print("✅ Stopped")