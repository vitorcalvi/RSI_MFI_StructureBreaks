import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.rsi_mfi_cloud import RSIMFICloudStrategy
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier

load_dotenv()

class TradeEngine:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.strategy = RSIMFICloudStrategy(self.risk_manager)
        self.notifier = TelegramNotifier()
        
        self.symbol = "ETHUSDT"
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
        self.position = None
        self.position_start_time = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.current_signal = None
        
        # Setup logging
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
        
        print(f"⚡ HF Trade engine initialized - {self.symbol}")
    
    def connect(self):
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Test connection
            info = self.exchange.get_server_time()
            if info.get('retCode') == 0:
                print(f"✅ Connected to Bybit {'testnet' if self.demo_mode else 'mainnet'}")
                return True
            
            print(f"❌ Connection failed: {info.get('retMsg')}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_quantity(self, qty):
        try:
            qty_step = "0.001"  # ETH precision
            decimals = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
            return f"{qty:.{decimals}f}" if decimals > 0 else str(int(qty))
        except:
            return f"{qty:.3f}"
    
    def log_trade(self, action, price, **kwargs):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if action == "ENTRY":
                self.trade_id += 1
                signal = kwargs.get('signal', {})
                qty = kwargs.get('quantity', '')
                side = signal.get('action', '')
                strategy = signal.get('signal_type', '')
                rsi = signal.get('rsi', 0)
                mfi = signal.get('mfi', 0)
                structure_stop = signal.get('structure_stop', 0)
                level = signal.get('level', 0)
                param_hash = kwargs.get('param_hash', '7e3d21a')
                
                log_line = (f"{timestamp}  id={self.trade_id}  action=ENTRY  side={side}  "
                           f"price={price:.2f}  size={qty}  strat={strategy}  rsi={rsi:.1f}  "
                           f"mfi={mfi:.1f}  structure_stop={structure_stop:.2f}  level={level:.2f}  "
                           f"param_hash={param_hash}  hold_s=0.0   pnl=0.00")
                
            elif action == "EXIT":
                duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                trigger = kwargs.get('reason', '').lower().replace(' ', '_')
                pnl = kwargs.get('pnl', 0)
                final = kwargs.get('final', 'true')
                param_hash = kwargs.get('param_hash', '7e3d21a')
                rsi = kwargs.get('rsi', 0)
                mfi = kwargs.get('mfi', 0)
                
                log_line = (f"{timestamp}  id={self.trade_id}  action=EXIT   trigger={trigger}  "
                           f"price={price:.2f}  pnl={pnl:.2f}  hold_s={duration:.1f}  final={final}  "
                           f"param_hash={param_hash}  rsi={rsi:.1f}  mfi={mfi:.1f}")
            
            with open(self.log_file, "a") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"Logging error: {e}")
    
    async def run_cycle(self):
        try:
            if not await self.update_market_data():
                return
            
            await self.check_position_status()
            
            # Check max hold time
            if self.position and self.position_start_time:
                age = (datetime.now() - self.position_start_time).total_seconds()
                if age > self.risk_manager.config['max_position_time']:
                    await self.close_position("max_hold_time_exceeded")
                    return
            
            signal = self.strategy.generate_signal(self.price_data)
            
            if signal and not self.position:
                await self.execute_trade(signal)
            elif self.position:
                await self.manage_position()
            
            self.display_status(signal)
        except Exception as e:
            print(f"❌ Cycle error: {e}")
    
    async def update_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
                limit=50
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
        except:
            return False
    
    async def check_position_status(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    await self.on_position_closed()
                self.position = None
                self.position_start_time = None
                self.current_signal = None
                return
            
            if not self.position:
                self.position_start_time = datetime.now()
            
            self.position = pos_list[0]
        except:
            pass
    
    async def execute_trade(self, signal):
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            balance = await self.get_account_balance()
            
            if not balance:
                return
            
            # Validate trade
            is_valid, reason = self.risk_manager.validate_trade(signal, balance, current_price)
            if not is_valid:
                print(f"⚠️  Trade rejected: {reason}")
                return
            
            # Calculate position size
            qty = self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=current_price,
                stop_price=signal['structure_stop']
            )
            
            formatted_qty = self.format_quantity(qty)
            if formatted_qty == "0":
                print("⚠️  Position size too small")
                return
            
            # Place order
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
                
                # Get current indicators for logging
                df = self.strategy.calculate_indicators(self.price_data)
                current_rsi = df.get('rsi', pd.Series([0])).iloc[-1] if len(df.get('rsi', [])) > 0 else 0
                current_mfi = df.get('mfi', pd.Series([0])).iloc[-1] if len(df.get('mfi', [])) > 0 else 0
                
                # Add indicators to signal for logging
                signal_with_indicators = signal.copy()
                signal_with_indicators['rsi'] = current_rsi
                signal_with_indicators['mfi'] = current_mfi
                
                self.log_trade("ENTRY", current_price, signal=signal_with_indicators, quantity=formatted_qty)
                await self.notifier.send_trade_signal(signal, current_price, formatted_qty)
                print(f"⚡ {signal['action']} order placed - Qty: {formatted_qty}")
            else:
                print(f"❌ Order failed: {order.get('retMsg')}")
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
    async def manage_position(self):
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
            age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
            
            should_close, reason = self.risk_manager.should_close_position(
                current_price=current_price,
                entry_price=float(self.position.get('avgPrice', 0)),
                side=self.position.get('side', ''),
                unrealized_pnl=unrealized_pnl,
                position_age_seconds=age
            )
            
            if should_close:
                await self.close_position(reason)
        except:
            pass
    
    async def close_position(self, reason="Manual"):
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
                # Get current indicators for logging
                df = self.strategy.calculate_indicators(self.price_data)
                current_rsi = df.get('rsi', pd.Series([0])).iloc[-1] if len(df.get('rsi', [])) > 0 else 0
                current_mfi = df.get('mfi', pd.Series([0])).iloc[-1] if len(df.get('mfi', [])) > 0 else 0
                
                self.log_trade("EXIT", current_price, reason=reason, pnl=pnl, 
                             rsi=current_rsi, mfi=current_mfi, final=True)
                print(f"⚡ Position closed - {reason}")
        except Exception as e:
            print(f"❌ Close error: {e}")
    
    async def get_account_balance(self):
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
            
            return 0
        except:
            return 0
    
    async def on_position_closed(self):
        try:
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
                
                # Get current indicators
                df = self.strategy.calculate_indicators(self.price_data)
                current_rsi = df.get('rsi', pd.Series([0])).iloc[-1] if len(df.get('rsi', [])) > 0 else 0
                current_mfi = df.get('mfi', pd.Series([0])).iloc[-1] if len(df.get('mfi', [])) > 0 else 0
                
                self.log_trade("EXIT", price, reason="position_closed", pnl=pnl,
                             rsi=current_rsi, mfi=current_mfi, final=True)
            
            await self.notifier.send_position_closed()
            print("📝 Position closed")
        except:
            pass
    
    def display_status(self, signal):
        try:
            price = float(self.price_data['close'].iloc[-1])
            time = self.price_data.index[-1].strftime('%H:%M:%S')
            
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print("⚡ ETHUSDT HIGH-FREQUENCY SCALPING BOT")
            print("=" * 60)
            print(f"⏰ {time} | 💰 ${price:.2f}")
            
            if signal:
                print(f"📊 Signal: {signal['action']} - {signal['signal_type']}")
                if 'rsi' in signal and 'mfi' in signal:
                    print(f"📈 RSI: {signal['rsi']} | MFI: {signal['mfi']}")
            else:
                if len(self.price_data) > 5:
                    df = self.strategy.calculate_indicators(self.price_data)
                    rsi = df.get('rsi', pd.Series([50])).iloc[-1]
                    mfi = df.get('mfi', pd.Series([50])).iloc[-1]
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
        except:
            pass