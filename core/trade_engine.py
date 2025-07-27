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
                # Live trading
                api_key = os.getenv('BYBIT_API_KEY', '').strip()
                api_secret = os.getenv('BYBIT_API_SECRET', '').strip()
                
                if not api_key or not api_secret:
                    raise ValueError("Live trading requires BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
                    
                self.exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                print("✅ Connected to Bybit Mainnet (LIVE TRADING)")
                
            try:
                self.exchange.load_markets()
                print(f"✅ Markets loaded successfully")
                
                # Test balance only if we have API keys
                if not self.demo_mode:
                    try:
                        balance = self.exchange.fetch_balance()
                        usdt_balance = balance.get('USDT', {}).get('free', 0)
                        print(f"💰 USDT Balance: ${usdt_balance:.2f}")
                    except Exception as e:
                        print(f"⚠️  Balance check failed: {str(e)[:50]}")
                else:
                    print("💡 Demo mode - using public market data only")
                        
            except Exception as e:
                print(f"⚠️  Exchange initialization error: {e}")
                
        else:
            raise ValueError(f"Exchange {self.exchange_name} not supported")
    
    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        """Fetch real OHLCV data"""
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
            
            # Validate data quality
            if len(df) < 20:
                print(f"⚠️  Insufficient data for {symbol}: {len(df)} candles")
                return None
                
            # Check for valid volume data
            zero_volume_count = (df['volume'] == 0).sum()
            if zero_volume_count > 0:
                print(f"⚠️  {zero_volume_count} zero volume bars in {symbol}")
                
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
        """Execute trade based on signal"""
        try:
            print(f"\n{'='*50}")
            print(f"🎯 SIGNAL: {signal['action']} {symbol}")
            print(f"Price: ${signal['price']:.4f}")
            print(f"RSI: {signal['rsi']:.2f} | MFI: {signal['mfi']:.2f}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Handle demo mode
            if self.demo_mode:
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
                    
                elif signal['action'] == 'SELL' and symbol in self.positions:
                    if self.positions[symbol]['side'] == 'long':
                        entry = self.positions[symbol]['entry_price']
                        pnl_pct = ((signal['price'] - entry) / entry) * 100
                        pnl_usd = pnl_pct * 10  # Demo calculation
                        
                        print(f"[DEMO] Position closed - P&L: {pnl_pct:+.2f}%")
                        await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                        
                        if pnl_pct > 0:
                            self.win_count += 1
                        else:
                            self.loss_count += 1
                            
                        del self.positions[symbol]
                        
                print(f"{'='*50}\n")
                return
            
            # Real trading logic
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 10:
                print("❌ Insufficient USDT balance for trading")
                return
            
            # Calculate position size
            position_value = self.risk_manager.calculate_position_size(usdt_balance, signal['price'])
            
            current_position = await self.get_current_position(symbol)
            
            if signal['action'] == 'BUY' and not current_position:
                # Place buy order
                order = self.exchange.create_market_order(
                    symbol, 
                    'buy', 
                    position_value
                )
                
                print(f"✅ BUY order executed")
                print(f"Order ID: {order['id']}")
                print(f"Amount: {position_value:.4f} {symbol.split('/')[0]}")
                
                self.trade_count += 1
                await self.telegram.trade_opened(symbol, signal['price'], position_value)
                
                # Set stop loss
                try:
                    sl_price = self.risk_manager.get_stop_loss(signal['price'], 'long')
                    sl_order = self.exchange.create_order(
                        symbol,
                        'stop_market',
                        'sell',
                        position_value,
                        None,
                        {'stopPrice': sl_price, 'reduceOnly': True}
                    )
                    print(f"🛡️ Stop loss set at ${sl_price:.4f}")
                except Exception as e:
                    print(f"⚠️  Stop loss failed: {e}")
                    
            elif signal['action'] == 'SELL' and current_position:
                # Close position
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
                else:
                    self.loss_count += 1
                    
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            print(f"{'='*50}\n")
    
    async def display_status(self, symbol, df):
        """Display real-time status"""
        try:
            df_with_indicators = self.strategy.calculate_indicators(df.copy())
            
            current_price = df['close'].iloc[-1]
            current_rsi = df_with_indicators['rsi'].iloc[-1]
            current_mfi = df_with_indicators['mfi'].iloc[-1]
            
            # Skip display if indicators are invalid
            if pd.isna(current_rsi) or pd.isna(current_mfi):
                return
                
            # Price movement
            if len(df) >= 5:
                price_change_5m = ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100
            else:
                price_change_5m = 0
            
            # Volume analysis
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Position info
            position_info = ""
            if symbol in self.positions:
                entry = self.positions[symbol]['entry_price']
                unrealized_pnl = ((current_price - entry) / entry) * 100
                position_info = f"| POS: {unrealized_pnl:+.2f}%"
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {symbol}: "
                  f"${current_price:.4f} | "
                  f"RSI: {current_rsi:.1f} | "
                  f"MFI: {current_mfi:.1f} | "
                  f"5m: {price_change_5m:+.2f}% | "
                  f"Vol: {volume_ratio:.1f}x "
                  f"{position_info} | "
                  f"T:{self.trade_count} W:{self.win_count} L:{self.loss_count}", end='')
            
        except Exception as e:
            print(f"\rStatus error: {str(e)[:30]}", end='')
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        mode_text = "DEMO" if self.demo_mode else "LIVE"
        print(f"\n🚀 Starting Trading Bot")
        print(f"Mode: {mode_text}")
        print(f"Exchange: {self.exchange_name.upper()}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Strategy: RSI+MFI ({self.strategy.params['rsi_length']}/{self.strategy.params['mfi_length']})")
        print(f"Risk: {self.risk_manager.max_position_size*100:.0f}% per trade, {self.risk_manager.stop_loss_pct*100:.1f}% stop loss")
        print("=" * 60)
        
        if not self.demo_mode:
            print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        else:
            print("💡 Demo mode - signals logged only (no real trades)")
            
        print("Monitoring for signals...\n")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Fetch real market data
                    df = await self.fetch_ohlcv(symbol, timeframe='1m', limit=100)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Display status
                    await self.display_status(symbol, df)
                    
                    # Check for signals
                    signal = self.strategy.generate_signal(df)
                    
                    if signal:
                        print()  # New line before signal
                        await self.execute_trade(signal, symbol)
                
                # Update frequency
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                await asyncio.sleep(10)
        
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