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
        
        # State - REMOVED: All cooldown references
        self.exchange = None
        self.running = False
        self.position = None
        self.profit_lock_active = False
    
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
            result = self.exchange.set_leverage(
                category="linear",
                symbol=self.linear,
                buyLeverage=str(self.risk_manager.leverage),
                sellLeverage=str(self.risk_manager.leverage)
            )
            if result.get('retCode') != 0:
                print(f"⚠️ Failed to set leverage: {result.get('retMsg')}")
                # Try to get current leverage
                positions = self.exchange.get_positions(category="linear", symbol=self.linear)
                if positions.get('retCode') == 0 and positions['result']['list']:
                    current_leverage = positions['result']['list'][0].get('leverage', 'Unknown')
                    print(f"⚠️ Current leverage: {current_leverage}x (wanted: {self.risk_manager.leverage}x)")
            else:
                print(f"✅ Leverage set to {self.risk_manager.leverage}x")
        except Exception as e:
            print(f"❌ Leverage setting error: {e}")
    
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
    
    def calculate_atr(self, df, period=14):
        """Calculate ATR for the dataframe"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().iloc[-1]
    
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
                    unrealized_pnl = float(position['unrealisedPnl'])
                    avg_price = float(position['avgPrice'])
                    leverage = float(position['leverage'])
                    liq_price = float(position.get('liqPrice', 0))
                    
                    # Calculate position metrics
                    position_value = position_size * avg_price
                    margin_used = position_value / leverage
                    
                    # FIXED: Calculate position P&L% manually (Bybit doesn't return unrealisedPnlPct)
                    # Position P&L% = unrealized_pnl / margin_used * 100
                    if margin_used > 0:
                        position_pnl_pct = (unrealized_pnl / margin_used) * 100
                    else:
                        position_pnl_pct = 0
                    
                    # Calculate wallet impact for display
                    wallet_balance = self.get_wallet_balance_only()
                    wallet_pnl_pct = (unrealized_pnl / wallet_balance) * 100 if wallet_balance > 0 else 0
                    
                    self.position = {
                        'side': position['side'],
                        'size': position_size,
                        'avg_price': avg_price,
                        'unrealized_pnl': unrealized_pnl,
                        'position_pnl_pct': position_pnl_pct,  # FIXED: Calculated manually
                        'wallet_pnl_pct': wallet_pnl_pct,     # FIXED: Direct wallet impact
                        'margin_used': margin_used,
                        'position_value': position_value,
                        'leverage': leverage,
                        'liq_price': liq_price
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
        if tick == 0:  # Add this check
            return str(price)
        price = round(price / tick) * tick
        
        tick_str = f"{tick:.20f}".rstrip('0').rstrip('.')
        decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{decimals}f}"

    async def handle_risk_management(self, current_price):
        """FIXED: Risk management with position P&L%"""
        if not self.position:
            return
            
        # FIXED: Use position P&L% for thresholds (already accounts for leverage)
        position_pnl_pct = self.position['position_pnl_pct']
        wallet_pnl_pct = self.position['wallet_pnl_pct']  # For display only
        
        # PRIORITY 1: Profit Protection (HIGHEST)
        if self.risk_manager.should_take_profit_protection(position_pnl_pct):
            print(f"\n💰 PROFIT PROTECTION: {position_pnl_pct:.1f}% pos ({wallet_pnl_pct:.2f}% wallet)")
            await self.close_position("Profit Protection")
            return
        
        # PRIORITY 2: Profit Lock (only if no protection)
        if not self.profit_lock_active and self.risk_manager.should_activate_profit_lock(position_pnl_pct):
            self.profit_lock_active = True
            print(f"\n🔓 PROFIT LOCK: {position_pnl_pct:.1f}% pos ({wallet_pnl_pct:.2f}% wallet)")
            
            await self.notifier.profit_lock_activated(
                self.symbol, wallet_pnl_pct, self.risk_manager.trailing_stop_distance * 100
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
        # Check if position already exists
        if self.position:
            print(f"⚠️ Position already exists, skipping new position")
            return False
        
        try:
            wallet_balance = self.get_wallet_balance_only()
            ticker = self.exchange.get_tickers(category="linear", symbol=self.linear)
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate position details
            position_size = self.risk_manager.calculate_position_size(wallet_balance, current_price)
            position_value = position_size * current_price
            required_margin = position_value / self.risk_manager.leverage
            
            # Safety check - don't use more than 50% of wallet for margin
            if required_margin > wallet_balance * 0.5:
                print(f"❌ Position too large! Required margin: ${required_margin:.2f}, Wallet: ${wallet_balance:.2f}")
                print(f"   Position value: ${position_value:.2f}, Leverage: {self.risk_manager.leverage}x")
                return False
            
            # Calculate approximate liquidation price
            # Liquidation = Entry × (1 - 1/Leverage + Maintenance Margin Rate)
            # Bybit maintenance margin is typically 0.5%
            maintenance_rate = 0.005
            if signal['action'] == 'BUY':
                approx_liq_price = current_price * (1 - 1/self.risk_manager.leverage + maintenance_rate)
            else:
                approx_liq_price = current_price * (1 + 1/self.risk_manager.leverage - maintenance_rate)
            
            liq_distance = abs(approx_liq_price - current_price) / current_price * 100
            
            print(f"\n📊 Position Analysis:")
            print(f"   Entry Price: ${current_price:.6f}")
            print(f"   Position Size: {position_size:.0f} tokens")
            print(f"   Position Value: ${position_value:.2f}")
            print(f"   Required Margin: ${required_margin:.2f} ({required_margin/wallet_balance*100:.1f}% of wallet)")
            print(f"   Approx Liquidation: ${approx_liq_price:.6f} ({liq_distance:.2f}% away)")
            
            # Warning if liquidation is too close
            if liq_distance < 5:
                print(f"⚠️ WARNING: Liquidation price is only {liq_distance:.2f}% away!")
                if liq_distance < 2:
                    print(f"❌ Position too risky, liquidation too close!")
                    return False
            
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
            
            # Set stop loss immediately
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
            
            # REMOVED: All cooldown logic
            
            wallet_pnl_pct = self.position.get('wallet_pnl_pct', 0)
            pnl = self.position.get('unrealized_pnl', 0)
            await self.notifier.trade_closed(self.symbol, wallet_pnl_pct, pnl, reason)
            
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
        """FIXED: Only reverse on loss using position P&L% - NO COOLDOWN"""
        current_side = self.position['side']
        position_pnl_pct = self.position['position_pnl_pct']  # FIXED: Use position P&L%
        signal_action = signal['action']
        
        # Check if signal is opposite
        should_reverse = (
            (current_side == 'Buy' and signal_action == 'SELL') or 
            (current_side == 'Sell' and signal_action == 'BUY')
        )
        
        if should_reverse:
            # FIXED: Only reverse on loss using position P&L% - NO COOLDOWN CHECK
            if self.risk_manager.should_reverse_for_loss(position_pnl_pct):
                wallet_pnl = self.position['wallet_pnl_pct']
                print(f"\n🔄 Reversing: {position_pnl_pct:.1f}% pos ({wallet_pnl:.2f}% wallet)")
                await self.close_position("Loss Reversal")
                await asyncio.sleep(1)
                await self.open_position(signal)
    
    async def _handle_signal_no_position(self, signal):
        # REMOVED: All cooldown logic
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
            
            # REMOVED: All cooldown handling
            
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
        
        # FIXED: Position info with position P&L% - REMOVED COOLDOWN
        position_info = 'None'
        if self.position:
            pos_pnl = self.position['position_pnl_pct']
            wallet_pnl = self.position['wallet_pnl_pct']
            liq_price = self.position.get('liq_price', 0)
            risk_zone = self.risk_manager.get_risk_zone(pos_pnl)
            zone_emoji = {
                'PROFIT_PROTECTION': '💰',
                'PROFIT_LOCK': '🔒',
                'LOSS_REVERSAL': '🔄',
                'NORMAL': ''
            }.get(risk_zone, '')
            
            # Calculate liquidation distance
            if liq_price > 0:
                liq_distance = abs(liq_price - current_price) / current_price * 100
                liq_info = f" Liq:{liq_distance:.1f}%"
            else:
                liq_info = ""
            
            position_info = f"{self.position['side']} ({pos_pnl:+.1f}%pos/{wallet_pnl:+.2f}%w{liq_info}) {zone_emoji}"
        
        # Trend display
        trend_emoji = {"UP": "🟢", "DOWN": "🔴", "SIDEWAYS": "🟡"}.get(current_trend, "⚪")
        
        # REMOVED: All cooldown display logic
        
        status = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"${current_price:.4f} | RSI:{current_rsi:.1f} | MFI:{current_mfi:.1f} | "
                  f"{trend_emoji} | {position_info}")
        
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