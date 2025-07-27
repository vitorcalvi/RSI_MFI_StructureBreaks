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
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.symbols = os.getenv('SYMBOLS', 'SOL/USDT').split(',')
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
            api_key = os.getenv('TESTNET_BYBIT_API_KEY' if self.demo_mode else 'BYBIT_API_KEY')
            api_secret = os.getenv('TESTNET_BYBIT_API_SECRET' if self.demo_mode else 'BYBIT_API_SECRET')
            
            api_key = api_key.strip() if api_key else ''
            api_secret = api_secret.strip() if api_secret else ''
            
            if self.demo_mode and (not api_key or not api_secret):
                print("⚠️  WARNING: No testnet API keys found")
                print("   To create testnet keys:")
                print("   1. Go to https://testnet.bybit.com")
                print("   2. Create a NEW account (separate from mainnet)")
                print("   3. Go to API Management")
                print("   4. Create API keys and add to .env file")
            
            if self.demo_mode:
                self.exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True
                    }
                })
                self.exchange.set_sandbox_mode(True)
                print("✅ Connected to Bybit Testnet (Public Mode)")
            else:
                if not api_key or not api_secret:
                    raise ValueError("Mainnet requires API credentials")
                    
                self.exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True
                    }
                })
                print("✅ Connected to Bybit Mainnet")
                
            try:
                self.exchange.load_markets()
                
                if api_key and api_secret and not self.demo_mode:
                    balance = self.exchange.fetch_balance()
                    usdt_balance = balance.get('USDT', {}).get('free', 0)
                    print(f"💰 USDT Balance: ${usdt_balance:.2f}")
            except Exception as e:
                if "API key" not in str(e):
                    print(f"⚠️  Connection test: {str(e)[:100]}")
                
        else:
            raise ValueError(f"Exchange {self.exchange_name} not supported")
    
    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        """Fetch OHLCV data with volume simulation for testnet"""
        try:
            # Get fresh 1-minute data
            since = self.exchange.milliseconds() - (limit * 60 * 1000)  # 1m * limit
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                print(f"No data returned for {symbol}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='last')]
            
            # Check for zero volume (common in testnet)
            zero_volume_count = (df['volume'] == 0).sum()
            if zero_volume_count > len(df) * 0.5:  # More than 50% zero volume
                print(f"⚠️  Low volume data detected ({zero_volume_count}/{len(df)} bars)")
                
                # Simulate realistic volume based on price movements for testnet
                if self.demo_mode:
                    price_changes = df['close'].pct_change().abs()
                    avg_price = df['close'].mean()
                    
                    # Generate volume based on price volatility
                    base_volume = avg_price * 100  # Base volume
                    df['volume'] = base_volume * (1 + price_changes * 20)
                    df['volume'] = df['volume'].fillna(base_volume)
                    
                    # Add some randomness
                    df['volume'] *= np.random.uniform(0.8, 1.2, size=len(df))
                    
                    print("✅ Simulated volume data for testnet")
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def get_ticker_info(self, symbol):
        """Get ticker information for scalping"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': ticker['ask'] - ticker['bid'],
                'spread_pct': ((ticker['ask'] - ticker['bid']) / ticker['bid']) * 100,
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage']
            }
        except:
            return None
    
    async def get_position(self, symbol):
        """Get current position for symbol"""
        if self.demo_mode:
            return None
            
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
        except:
            return None
    
    async def execute_trade(self, signal, symbol):
        """Execute trade based on signal"""
        try:
            print(f"\n{'='*60}")
            print(f"🎯 SCALPING SIGNAL: {signal['action']}")
            print(f"Symbol: {symbol}")
            print(f"Price: ${signal['price']:.4f}")
            print(f"RSI: {signal['rsi']:.2f}")
            print(f"MFI: {signal['mfi']:.2f}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            if self.demo_mode:
                print("[DEMO MODE] Signal logged only")
                
                if signal['action'] == 'BUY' and symbol not in self.positions:
                    self.positions[symbol] = {
                        'entry_price': signal['price'],
                        'entry_time': datetime.now()
                    }
                    self.trade_count += 1
                    # Demo mode notification
                    await self.telegram.trade_opened(symbol, signal['price'], 0.1)
                elif signal['action'] == 'SELL' and symbol in self.positions:
                    entry = self.positions[symbol]['entry_price']
                    pnl_pct = ((signal['price'] - entry) / entry) * 100
                    pnl_usd = pnl_pct * 10  # Assuming $1000 demo position
                    print(f"[DEMO] Position closed - P&L: {pnl_pct:+.2f}%")
                    
                    # Demo mode notification
                    await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                    
                    if pnl_pct > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                        
                    del self.positions[symbol]
                    
                print(f"{'='*60}\n")
                return
            
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 10:
                print("❌ Insufficient balance")
                return
            
            position_value = self.risk_manager.calculate_position_size(usdt_balance, signal['price'])
            position_size = position_value / signal['price']
            
            current_position = await self.get_position(symbol)
            
            if signal['action'] == 'BUY':
                if not current_position:
                    order = self.exchange.create_market_order(
                        symbol, 
                        'buy', 
                        position_size
                    )
                    
                    print(f"✅ Long position opened")
                    print(f"Order ID: {order['id']}")
                    print(f"Size: {position_size:.4f} {symbol.split('/')[0]}")
                    self.trade_count += 1
                    
                    # Send Telegram notification
                    await self.telegram.trade_opened(symbol, signal['price'], position_size)
                    
                    # Tighter stop loss for scalping
                    sl_price = signal['price'] * 0.995  # 0.5% stop loss
                    try:
                        sl_order = self.exchange.create_order(
                            symbol,
                            'stop',
                            'sell',
                            position_size,
                            None,
                            {'stopPrice': sl_price, 'reduceOnly': True}
                        )
                        print(f"🛡️ Stop loss set at ${sl_price:.4f}")
                    except:
                        print("⚠️  Could not set stop loss")
                        
                    self.positions[symbol] = {
                        'entry_price': signal['price'],
                        'entry_time': datetime.now(),
                        'size': position_size
                    }
                    
            elif signal['action'] == 'SELL':
                if current_position and current_position['side'] == 'long':
                    order = self.exchange.create_market_order(
                        symbol,
                        'sell',
                        current_position['contracts'],
                        params={'reduceOnly': True}
                    )
                    
                    pnl_pct = current_position['pnl_pct']
                    print(f"✅ Position closed")
                    print(f"P&L: {pnl_pct:+.2f}% (${current_position['pnl']:.2f})")
                    
                    # Send Telegram notification
                    await self.telegram.trade_closed(symbol, pnl_pct, current_position['pnl'])
                    
                    if pnl_pct > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                    
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            print(f"{'='*60}\n")
    
    async def display_status(self, symbol, df, ticker_info):
        """Display scalping status"""
        try:
            # Get indicators from strategy calculation
            df_with_indicators = self.strategy.calculate_indicators(df.copy())
            
            current_price = df['close'].iloc[-1]
            current_rsi = df_with_indicators['rsi'].iloc[-1]
            current_mfi = df_with_indicators['mfi'].iloc[-1]
            
            # Price movement (last 5 candles)
            price_change_5m = ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100
            
            # Volume info
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Clear line and display info
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {symbol}: "
                  f"${current_price:.4f} | "
                  f"RSI: {current_rsi:.1f} | "
                  f"MFI: {current_mfi:.1f} | "
                  f"5m: {price_change_5m:+.2f}% | "
                  f"Vol: {volume_ratio:.1f}x | "
                  f"Spread: {ticker_info['spread_pct']:.3f}% | "
                  f"Trades: {self.trade_count} W:{self.win_count} L:{self.loss_count}", end='')
            
        except Exception as e:
            print(f"\rError displaying status: {str(e)[:50]}", end='')
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        print(f"\n🚀 Starting 1-Minute Scalping Bot")
        print(f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"Exchange: {self.exchange_name.upper()}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"\nScalping Parameters:")
        print(f"├─ Timeframe: 1 minute")
        print(f"├─ RSI Period: {self.strategy.params['rsi_length']}")
        print(f"├─ Oversold: {self.strategy.params['oversold_level']}")
        print(f"├─ Overbought: {self.strategy.params['overbought_level']}")
        print(f"├─ MFI Period: {self.strategy.params['mfi_length']}")
        print(f"├─ Stop Loss: 0.5%")
        print(f"└─ Risk per trade: {self.risk_manager.max_position_size*100:.0f}%")
        print("=" * 60)
        print("\nMonitoring for scalping opportunities...")
        
        iteration = 0
        while self.running:
            try:
                iteration += 1
                
                for symbol in self.symbols:
                    # Fetch 1-minute data
                    df = await self.fetch_ohlcv(symbol, timeframe='1m', limit=100)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Get ticker info
                    ticker_info = await self.get_ticker_info(symbol)
                    
                    # Display real-time status
                    if ticker_info:
                        await self.display_status(symbol, df, ticker_info)
                    
                    # Check for signals
                    signal = self.strategy.generate_signal(df)
                    
                    if signal:
                        print()  # New line before signal
                        await self.execute_trade(signal, symbol)
                    
                    # Show open positions
                    if not self.demo_mode and iteration % 10 == 0:
                        position = await self.get_position(symbol)
                        if position:
                            print(f"\n📈 Open: {position['side']} {position['contracts']:.4f} @ ${position['avg_price']:.4f}, P&L: {position['pnl_pct']:+.2f}%")
                
                # Scalping requires faster updates
                await asyncio.sleep(5)  # Check every 5 seconds for 1-minute scalping
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                await asyncio.sleep(5)
        
        await self.stop()
    
    async def stop(self):
        """Stop trading engine"""
        self.running = False
        
        print(f"\n\n🛑 Stopping scalping bot...")
        print(f"📊 Final Stats: Trades: {self.trade_count}, Wins: {self.win_count}, Losses: {self.loss_count}")
        
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            print(f"📈 Win Rate: {win_rate:.1f}%")
        
        if not self.demo_mode:
            for symbol in self.symbols:
                try:
                    position = await self.get_position(symbol)
                    if position:
                        print(f"Closing position on {symbol}...")
                        
                        order = self.exchange.create_market_order(
                            symbol,
                            'sell' if position['side'] == 'long' else 'buy',
                            position['contracts'],
                            params={'reduceOnly': True}
                        )
                        print(f"✅ Position closed: {order['id']}")
                except Exception as e:
                    print(f"❌ Error closing position: {e}")
        
        print("✅ Scalping bot stopped")