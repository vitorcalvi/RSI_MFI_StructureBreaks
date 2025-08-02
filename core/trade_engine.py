import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.rsi_mfi_strategy import RSIMFIStrategy
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier

load_dotenv()

class TradeEngine:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.strategy = RSIMFIStrategy()
        self.notifier = TelegramNotifier()
        
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API credentials
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        # State variables
        self.exchange = None
        self.position = None
        self.position_start_time = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.current_signal = None
        
        # Symbol-specific trading rules (will be loaded on connect)
        self.qty_step = "0.001"  # Default, will be updated
        self.min_qty = 0.001     # Default, will be updated
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
        
        print(f"⚡ Trade Engine initialized - {self.symbol}")
    
    def connect(self):
        """Connect to Bybit exchange and load symbol info"""
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            info = self.exchange.get_server_time()
            if info.get('retCode') != 0:
                print(f"❌ Connection failed: {info.get('retMsg')}")
                return False
            
            # Load symbol-specific trading rules
            self._load_symbol_info()
            
            mode = 'testnet' if self.demo_mode else 'mainnet'
            print(f"✅ Connected to Bybit {mode}")
            print(f"📊 {self.symbol} - Qty Step: {self.qty_step}, Min Qty: {self.min_qty}")
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _load_symbol_info(self):
        """Load symbol-specific trading rules from Bybit"""
        try:
            instruments = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            
            if instruments.get('retCode') == 0 and instruments['result']['list']:
                symbol_info = instruments['result']['list'][0]
                
                # Extract lot size info
                lot_size_filter = symbol_info.get('lotSizeFilter', {})
                self.qty_step = lot_size_filter.get('qtyStep', '0.001')
                self.min_qty = float(lot_size_filter.get('minOrderQty', '0.001'))
                
                print(f"✅ Loaded {self.symbol} rules: qtyStep={self.qty_step}, minQty={self.min_qty}")
            else:
                print(f"⚠️ Could not load {self.symbol} info, using defaults")
                # Set common defaults based on symbol
                if 'ETH' in self.symbol:
                    self.qty_step = '0.01'
                    self.min_qty = 0.01
                elif 'BTC' in self.symbol:
                    self.qty_step = '0.001'
                    self.min_qty = 0.001
                else:  # ADA and most others
                    self.qty_step = '1'
                    self.min_qty = 1.0
                    
        except Exception as e:
            print(f"❌ Symbol info load error: {e}")
            # Fallback defaults
            if 'ETH' in self.symbol:
                self.qty_step = '0.01'
                self.min_qty = 0.01
            elif 'BTC' in self.symbol:
                self.qty_step = '0.001'
                self.min_qty = 0.001  
            else:
                self.qty_step = '1'
                self.min_qty = 1.0
    
    def format_quantity(self, qty):
        """Format quantity according to exchange rules for the specific symbol"""
        try:
            if qty < self.min_qty:
                return "0"
            
            # Calculate decimal places from qty_step
            if '.' in self.qty_step:
                decimals = len(self.qty_step.split('.')[1])
            else:
                decimals = 0
            
            # Round to qty_step precision
            qty_step_float = float(self.qty_step)
            rounded_qty = round(qty / qty_step_float) * qty_step_float
            
            # Format with correct decimal places
            if decimals > 0:
                return f"{rounded_qty:.{decimals}f}"
            else:
                return str(int(rounded_qty))
                
        except Exception as e:
            print(f"❌ Quantity formatting error: {e}")
            return f"{qty:.3f}"  # Fallback
    
    def _log_trade(self, action, price, **kwargs):
        """Log trade to file with deduplication"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            qty = kwargs.get('quantity', '')
            log_line = (f"{timestamp}  id={self.trade_id}  action=ENTRY  side={signal.get('action', '')}  "
                       f"price={price:.2f}  size={qty}  strat={signal.get('signal_type', '')}  "
                       f"rsi={signal.get('rsi', 0):.1f}  mfi={signal.get('mfi', 0):.1f}  "
                       f"structure_stop={signal.get('structure_stop', 0):.2f}  level={signal.get('level', 0):.2f}  "
                       f"param_hash=7e3d21a  hold_s=0.0   pnl=0.00")
        else:  # EXIT
            duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
            log_line = (f"{timestamp}  id={self.trade_id}  action=EXIT   trigger={kwargs.get('reason', '').lower().replace(' ', '_')}  "
                       f"price={price:.2f}  pnl={kwargs.get('pnl', 0):.2f}  hold_s={duration:.1f}  final=true  "
                       f"param_hash=7e3d21a  rsi={kwargs.get('rsi', 0):.1f}  mfi={kwargs.get('mfi', 0):.1f}")
        
        try:
            # Check for duplicate EXIT records before writing
            if action == "EXIT" and self._is_duplicate_exit(log_line):
                return
            
            with open(self.log_file, "a") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    def _is_duplicate_exit(self, new_log_line):
        """Check if EXIT record is duplicate of recent entries"""
        try:
            if not os.path.exists(self.log_file):
                return False
            
            # Extract key fields from new log line
            import re
            new_id = re.search(r'id=(\d+)', new_log_line)
            new_price = re.search(r'price=([0-9.]+)', new_log_line)
            new_pnl = re.search(r'pnl=([0-9.-]+)', new_log_line)
            
            if not all([new_id, new_price, new_pnl]):
                return False
            
            new_id = int(new_id.group(1))
            new_price = float(new_price.group(1))
            new_pnl = float(new_pnl.group(1))
            
            # Read last 3 lines to check for duplicates
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 1:
                return False
            
            # Check last 2 lines for matching EXIT records
            for line in lines[-2:]:
                if 'action=EXIT' not in line:
                    continue
                
                existing_id = re.search(r'id=(\d+)', line)
                existing_price = re.search(r'price=([0-9.]+)', line)
                existing_pnl = re.search(r'pnl=([0-9.-]+)', line)
                
                if not all([existing_id, existing_price, existing_pnl]):
                    continue
                
                existing_id = int(existing_id.group(1))
                existing_price = float(existing_price.group(1))
                existing_pnl = float(existing_pnl.group(1))
                
                # Check if this is a duplicate (same id, price, pnl)
                if (new_id == existing_id and 
                    abs(new_price - existing_price) < 0.01 and 
                    abs(new_pnl - existing_pnl) < 0.01):
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Duplicate check error: {e}")
            return False
    
    async def run_cycle(self):
        """Run one trading cycle"""
        try:
            if not await self._update_market_data():
                return
            
            await self._check_position_status()
            
            if self.position and self.position_start_time:
                await self._check_position_exit()
            
            if not self.position:
                signal = self.strategy.generate_signal(self.price_data)
                if signal:
                    await self._execute_trade(signal)
            
            self._display_status()
            
        except Exception as e:
            print(f"❌ Cycle error: {e}")
    
    async def _update_market_data(self):
        """Fetch and update market data"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
                limit=200
            )
            
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').set_index('timestamp')
            return True
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def _check_position_status(self):
        """Check current position status from exchange"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    await self._on_position_closed()
                self.position = None
                self.position_start_time = None
                self.current_signal = None
                return
            
            if not self.position:
                self.position_start_time = datetime.now()
            
            self.position = pos_list[0]
            
        except Exception as e:
            print(f"❌ Position status error: {e}")
    
    async def _check_position_exit(self):
        """Check if position should be closed based on risk management"""
        try:
            if not self.position or not self.position_start_time:
                return
            
            current_price = float(self.price_data['close'].iloc[-1])
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
            position_age = (datetime.now() - self.position_start_time).total_seconds()
            
            should_close, reason = self.risk_manager.should_close_position(
                current_price=current_price,
                entry_price=entry_price,
                side=side,
                unrealized_pnl=unrealized_pnl,
                position_age_seconds=position_age
            )
            
            if should_close:
                await self._close_position(reason)
                
        except Exception as e:
            print(f"❌ Position exit check error: {e}")
    
    async def _execute_trade(self, signal):
        """Execute trade based on signal"""
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            balance = await self.get_account_balance()
            
            if not balance:
                print("⚠️ No balance available")
                return
            
            is_valid, reason = self.risk_manager.validate_trade(signal, balance, current_price)
            if not is_valid:
                print(f"⚠️ Trade rejected: {reason}")
                return
            
            qty = self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=current_price,
                stop_price=signal['structure_stop']
            )
            
            formatted_qty = self.format_quantity(qty)
            if formatted_qty == "0" or float(formatted_qty) < 0.001:
                print("⚠️ Position size too small")
                return
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Market",
                qty=formatted_qty,
                timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                self.current_signal = signal
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty)
                
                strategy_info = self.strategy.get_strategy_info()
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, strategy_info)
                
                print(f"⚡ {signal['action']} order placed - Qty: {formatted_qty}")
            else:
                print(f"❌ Order failed: {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
    async def _close_position(self, reason="Manual"):
        """Close current position"""
        try:
            if not self.position:
                return
            
            current_price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            side = "Sell" if self.position.get('side') == "Buy" else "Buy"
            qty = self.format_quantity(float(self.position['size']))
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=qty,
                timeInForce="IOC",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                indicators = self.strategy.calculate_indicators(self.price_data)
                current_rsi = indicators.get('rsi', pd.Series([0])).iloc[-1] if 'rsi' in indicators else 0
                current_mfi = indicators.get('mfi', pd.Series([0])).iloc[-1] if 'mfi' in indicators else 0
                
                self._log_trade("EXIT", current_price, reason=reason, pnl=pnl, 
                               rsi=current_rsi, mfi=current_mfi)
                
                exit_data = {
                    'trigger': reason,
                    'rsi': current_rsi,
                    'mfi': current_mfi
                }
                strategy_info = self.strategy.get_strategy_info()
                await self.notifier.send_trade_exit(exit_data, current_price, pnl, duration, strategy_info)
                
                print(f"⚡ Position closed - {reason}")
            else:
                print(f"❌ Close order failed: {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Close position error: {e}")
    
    async def get_account_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
            
            return 0
        except Exception as e:
            print(f"❌ Balance error: {e}")
            return 0
    
    async def _on_position_closed(self):
        """Handle position closed event"""
        try:
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
                
                indicators = self.strategy.calculate_indicators(self.price_data)
                current_rsi = indicators.get('rsi', pd.Series([0])).iloc[-1] if 'rsi' in indicators else 0
                current_mfi = indicators.get('mfi', pd.Series([0])).iloc[-1] if 'mfi' in indicators else 0
                
                self._log_trade("EXIT", price, reason="position_closed", pnl=pnl,
                               rsi=current_rsi, mfi=current_mfi)
            
            print("📝 Position closed externally")
        except Exception as e:
            print(f"❌ Position closed handler error: {e}")
    
    def _display_status(self):
        """Display current trading status"""
        try:
            price = float(self.price_data['close'].iloc[-1])
            time = self.price_data.index[-1].strftime('%H:%M:%S')
            
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print(f"⚡ {self.symbol} HIGH-FREQUENCY SCALPING BOT")
            print("=" * 60)
            print(f"⏰ {time} | 💰 ${price:.2f}")
            
            if len(self.price_data) > 10:
                indicators = self.strategy.calculate_indicators(self.price_data)
                if indicators:
                    rsi = indicators.get('rsi', pd.Series([50])).iloc[-1]
                    mfi = indicators.get('mfi', pd.Series([50])).iloc[-1]
                    print(f"📈 RSI: {rsi:.1f} | MFI: {mfi:.1f}")
            
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry = float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                pnl_pct = (pnl / (float(size) * entry)) * 100 if entry > 0 and size != '0' else 0
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                max_hold = self.risk_manager.config['max_position_time']
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"\n{emoji} POSITION: {side} | {size}")
                print(f"Entry: ${entry:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                print(f"⏱️ {age:.1f}s / {max_hold}s")
            else:
                print("\n⚡ No Position - Scanning...")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")