import os
import json
import ccxt
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
from core.risk_management import RiskManager
from core.telegram_notifier import TelegramNotifier

class TradeEngine:
    def __init__(self):
        self.demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
        self.symbols = [s.strip() for s in os.getenv('SYMBOLS', 'SOL/USDT').split(',')]
        self.exchange_name = os.getenv('EXCHANGE', 'bybit')
        self.positions = {}
        self.running = False
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        self._init_exchange()
        self.strategy = RSIMFICloudStrategy()
        self.risk_manager = RiskManager()
        self.telegram = TelegramNotifier()
        
    def _init_exchange(self):
        """Initialize exchange connection"""
        if self.exchange_name == 'bybit':
            if self.demo_mode:
                # Demo mode - use public endpoints only
                self.exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                print("✅ Connected to Bybit (Demo Mode - Public Data Only)")
            else:
                # TESTNET trading
                api_key = os.getenv('BYBIT_API_KEY', '').strip()
                api_secret = os.getenv('BYBIT_API_SECRET', '').strip()
                
                if not api_key or not api_secret:
                    raise ValueError("Testnet trading requires BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
                    
                self.exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'sandbox': True,  # TESTNET MODE
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                print("✅ Connected to Bybit TESTNET (SAFE REAL TRADING)")
                
            try:
                self.exchange.load_markets()
                print(f"✅ Markets loaded successfully")
                
                # Test balance only if we have API keys
                if not self.demo_mode:
                    try:
                        balance = self.exchange.fetch_balance()
                        usdt_balance = balance.get('USDT', {}).get('free', 0)
                        print(f"💰 TESTNET USDT Balance: ${usdt_balance:.2f}")
                    except Exception as e:
                        print(f"⚠️  Balance check failed: {str(e)[:50]}")
                else:
                    print("💡 Demo mode - using public market data only")
                        
            except Exception as e:
                print(f"⚠️  Exchange initialization error: {e}")
                
        else:
            raise ValueError(f"Exchange {self.exchange_name} not supported")
    
    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=200):
        """Fetch real OHLCV data with enhanced validation"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                print(f"❌ No data returned for {symbol}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # CRITICAL: Data validation for stress testing
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    print(f"❌ Invalid prices in {symbol}")
                    return None
            
            # Check for gaps > 3 minutes (for 1m data)
            time_diff = df.index.to_series().diff()
            max_gap = time_diff.max().total_seconds() / 60
            if max_gap > 5:
                print(f"⚠️ Data gap: {max_gap:.1f}min in {symbol}")
            
            # Remove extreme outliers (bad ticks)
            for col in price_cols:
                median_price = df[col].median()
                mask = (df[col] > median_price * 0.7) & (df[col] < median_price * 1.3)
                if not mask.all():
                    removed = (~mask).sum()
                    if removed > 0:
                        print(f"⚠️ Removed {removed} outliers from {symbol}")
                        df = df[mask]
            
            # Validate minimum data for indicators
            if len(df) < 30:
                print(f"❌ Insufficient valid data for {symbol}: {len(df)} bars")
                return None
                
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    async def get_current_position(self, symbol):
        """Get current position for symbol"""
        if self.demo_mode:
            # Demo mode - return stored position
            return self.positions.get(symbol)
            
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for pos in positions:
                if pos['contracts'] > 0:
                    return {
                        'side': pos['side'],
                        'contracts': pos['contracts'],
                        'avg_price': pos['average'],
                        'pnl': pos['unrealizedPnl'],
                        'pnl_pct': pos['percentage']
                    }
            return None
            
        except Exception as e:
            print(f"⚠️  Position check error: {e}")
            return None
    
    async def execute_trade(self, signal, symbol):
        """Execute trade based on signal - TESTNET REAL TRADING"""
        try:
            print(f"\n{'='*50}")
            print(f"🎯 SIGNAL: {signal['action']} {symbol}")
            print(f"Price: ${signal['price']:.4f}")
            print(f"RSI: {signal['rsi']:.2f} | MFI: {signal['mfi']:.2f}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Handle demo mode
            if self.demo_mode:
                print("Mode: DEMO")
                print("[DEMO MODE] Signal logged only")
                
                if signal['action'] == 'BUY' and symbol not in self.positions:
                    self.positions[symbol] = {
                        'entry_price': signal['price'],
                        'entry_time': datetime.now(),
                        'side': 'long',
                        'size': 0.1  # Demo size
                    }
                    self.trade_count += 1
                    await self.telegram.trade_opened(symbol, signal['price'], 0.1)
                    print(f"📈 Position OPENED - Total trades: {self.trade_count}")
                    
                elif signal['action'] == 'SELL' and symbol in self.positions:
                    if self.positions[symbol]['side'] == 'long':
                        entry = self.positions[symbol]['entry_price']
                        pnl_pct = ((signal['price'] - entry) / entry) * 100
                        pnl_usd = pnl_pct * 10  # Demo calculation
                        
                        print(f"📉 Position CLOSED - P&L: {pnl_pct:+.2f}%")
                        await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                        
                        if pnl_pct > 0:
                            self.win_count += 1
                            print("✅ WIN")
                        else:
                            self.loss_count += 1
                            print("❌ LOSS")
                            
                        del self.positions[symbol]
                        
                print(f"{'='*50}\n")
                return
            
            # TESTNET REAL TRADING
            print("Mode: TESTNET REAL TRADING")
            
            # Get balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 5:
                print("❌ Insufficient TESTNET USDT balance for trading")
                print(f"{'='*50}\n")
                return
            
            # Calculate position size (conservative for testnet)
            position_value = min(usdt_balance * 0.05, 10)  # Max $10 per trade
            position_size = position_value / signal['price']
            
            current_position = await self.get_current_position(symbol)
            
            if signal['action'] == 'BUY' and not current_position:
                # Place buy order
                print(f"💰 Placing BUY order: ${position_value:.2f} worth")
                
                order = self.exchange.create_market_order(
                    symbol, 
                    'buy', 
                    position_size
                )
                
                print(f"✅ BUY order executed")
                print(f"Order ID: {order['id']}")
                print(f"Amount: {position_size:.6f} {symbol.split('/')[0]}")
                
                self.trade_count += 1
                await self.telegram.trade_opened(symbol, signal['price'], position_size)
                
            elif signal['action'] == 'SELL' and current_position:
                # Close position
                print(f"💰 Closing position: {current_position['contracts']:.6f}")
                
                order = self.exchange.create_market_order(
                    symbol,
                    'sell',
                    current_position['contracts'],
                    params={'reduceOnly': True}
                )
                
                pnl_pct = current_position.get('pnl_pct', 0)
                pnl_usd = current_position.get('pnl', 0)
                
                print(f"✅ Position closed")
                print(f"Order ID: {order['id']}")
                print(f"P&L: {pnl_pct:+.2f}% (${pnl_usd:.2f})")
                
                await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                
                if pnl_pct > 0:
                    self.win_count += 1
                    print("✅ WIN")
                else:
                    self.loss_count += 1
                    print("❌ LOSS")
                    
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            print(f"{'='*50}\n")
    
    async def display_status(self, symbol, df):
        """Display real-time status with enhanced info"""
        try:
            df_with_indicators = self.strategy.calculate_indicators(df.copy())
            
            current_price = df['close'].iloc[-1]
            current_rsi = df_with_indicators['rsi'].iloc[-1]
            current_mfi = df_with_indicators['mfi'].iloc[-1]
            
            # Skip display if indicators are invalid
            if pd.isna(current_rsi) or pd.isna(current_mfi):
                return
                
            # Price movement (1 minute change for stress test)
            if len(df) >= 2:
                price_change_1m = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            else:
                price_change_1m = 0
            
            # Volume analysis
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Position info
            position_info = ""
            if self.demo_mode and symbol in self.positions:
                entry = self.positions[symbol]['entry_price']
                unrealized_pnl = ((current_price - entry) / entry) * 100
                position_info = f"| POS: {unrealized_pnl:+.2f}%"
            elif not self.demo_mode:
                # Check real position
                current_position = await self.get_current_position(symbol)
                if current_position:
                    pnl_pct = current_position.get('pnl_pct', 0)
                    position_info = f"| POS: {pnl_pct:+.2f}%"
            
            # Signal indicators
            signal_status = ""
            if current_rsi < 45 and current_mfi < 45:
                signal_status = "🟢BUY_ZONE"
            elif current_rsi > 55 and current_mfi > 55:
                signal_status = "🔴SELL_ZONE"
            else:
                signal_status = "⚪NEUTRAL"
            
            mode_indicator = "DEMO" if self.demo_mode else "TESTNET"
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {symbol}: "
                  f"${current_price:.4f} | "
                  f"RSI: {current_rsi:.1f} | "
                  f"MFI: {current_mfi:.1f} | "
                  f"1m: {price_change_1m:+.2f}% | "
                  f"Vol: {volume_ratio:.1f}x | "
                  f"{signal_status} "
                  f"{position_info} | "
                  f"{mode_indicator} | "
                  f"T:{self.trade_count} W:{self.win_count} L:{self.loss_count}", end='')
            
        except Exception as e:
            print(f"\rStatus error: {str(e)[:30]}", end='')
    
    async def run(self):
        """Main trading loop - TESTNET REAL TRADING"""
        self.running = True
        
        mode_text = "DEMO" if self.demo_mode else "TESTNET LIVE"
        print(f"\n🚀 Starting Trading Bot")
        print(f"Mode: {mode_text}")
        print(f"Exchange: {self.exchange_name.upper()}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: 1m")
        print(f"Strategy: RSI+MFI ({self.strategy.params['rsi_length']}/{self.strategy.params['mfi_length']})")
        print(f"Thresholds: {self.strategy.params['oversold_level']}/{self.strategy.params['overbought_level']}")
        print(f"Risk: {self.risk_manager.max_position_size*100:.0f}% per trade, {self.risk_manager.stop_loss_pct*100:.1f}% stop loss")
        print("=" * 60)
        
        if not self.demo_mode:
            print("🚀 TESTNET LIVE TRADING MODE - REAL ORDERS ON TESTNET")
        else:
            print("💡 Demo mode - signals logged only (no real trades)")
            
        print("Monitoring for signals...\n")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Fetch 1-minute data with rate limiting protection
                    df = await self.fetch_ohlcv(symbol, timeframe='1m', limit=200)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Display status
                    await self.display_status(symbol, df)
                    
                    # Check for signals
                    signal = self.strategy.generate_signal(df)
                    
                    if signal:
                        print()  # New line before signal
                        await self.execute_trade(signal, symbol)
                
                # Rate limiting protection: slower polling to avoid API limits
                await asyncio.sleep(8)  # Increased from 3 to 8 seconds
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                await asyncio.sleep(8)
        
        await self.stop()
    
    async def stop(self):
        """Stop trading engine"""
        self.running = False
        
        print(f"\n\n🛑 Stopping bot...")
        print(f"📊 Final Stats: Trades: {self.trade_count}, Wins: {self.win_count}, Losses: {self.loss_count}")
        
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            print(f"📈 Win Rate: {win_rate:.1f}%")
        
        print("✅ Bot stopped")