import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pybit.unified_trading import HTTP
from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
from core.risk_management import RiskManager
from core.telegram_notifier import TelegramNotifier

class TradeEngine:
    def __init__(self):
        self.demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
        self.symbols = [s.strip() for s in os.getenv('SYMBOLS', 'SOL/USDT').split(',')]
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
        """Initialize pybit connection"""
        try:
            if self.demo_mode:
                # Demo mode - use testnet
                api_key = os.getenv('TESTNET_BYBIT_API_KEY', '').strip()
                api_secret = os.getenv('TESTNET_BYBIT_API_SECRET', '').strip()
                
                self.session = HTTP(
                    testnet=True,
                    api_key=api_key if api_key else None,
                    api_secret=api_secret if api_secret else None
                )
                print("✅ Connected to Bybit Testnet")
                
            else:
                # Live trading
                api_key = os.getenv('BYBIT_API_KEY', '').strip()
                api_secret = os.getenv('BYBIT_API_SECRET', '').strip()
                
                if not api_key or not api_secret:
                    raise ValueError("Live trading requires BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
                    
                self.session = HTTP(
                    testnet=False,
                    api_key=api_key,
                    api_secret=api_secret
                )
                print("✅ Connected to Bybit Mainnet (LIVE TRADING)")
            
            # Test connection if we have API keys
            if not self.demo_mode or (os.getenv('TESTNET_BYBIT_API_KEY')):
                try:
                    balance = self.session.get_wallet_balance(accountType="UNIFIED")
                    if balance['retCode'] == 0:
                        usdt_balance = 0
                        for coin in balance['result']['list'][0]['coin']:
                            if coin['coin'] == 'USDT':
                                usdt_balance = float(coin['walletBalance'])
                                break
                        print(f"💰 USDT Balance: ${usdt_balance:.2f}")
                    else:
                        print(f"⚠️  Balance check failed: {balance['retMsg']}")
                except Exception as e:
                    print(f"⚠️  Balance check error: {str(e)[:50]}")
            else:
                print("💡 Demo mode - public data only")
                        
        except Exception as e:
            print(f"❌ Exchange initialization error: {e}")
            # Fallback to public-only session
            self.session = HTTP(testnet=self.demo_mode)
    
    async def fetch_ohlcv(self, symbol, timeframe='1', limit=100):
        """Fetch OHLCV data using pybit"""
        try:
            # Convert symbol format (SOL/USDT -> SOLUSDT)
            pybit_symbol = symbol.replace('/', '')
            
            # Convert timeframe format
            interval_map = {
                '1m': '1', '1': '1',
                '5m': '5', '5': '5',
                '15m': '15', '15': '15',
                '1h': '60', '60': '60',
                '4h': '240', '240': '240',
                '1d': 'D', 'D': 'D'
            }
            interval = interval_map.get(timeframe, '1')
            
            # Fetch kline data
            response = self.session.get_kline(
                category="spot",
                symbol=pybit_symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                print(f"❌ API error for {symbol}: {response['retMsg']}")
                return None
                
            klines = response['result']['list']
            if not klines:
                print(f"❌ No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for kline in reversed(klines):  # Reverse to get chronological order
                df_data.append([
                    int(kline[0]),      # timestamp
                    float(kline[1]),    # open
                    float(kline[2]),    # high
                    float(kline[3]),    # low
                    float(kline[4]),    # close
                    float(kline[5])     # volume
                ])
            
            df = pd.DataFrame(df_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # Validate data
            if len(df) < 20:
                print(f"⚠️  Insufficient data for {symbol}: {len(df)} candles")
                return None
                
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    async def get_current_position(self, symbol):
        """Get current position using pybit"""
        if self.demo_mode:
            return self.positions.get(symbol)
            
        try:
            pybit_symbol = symbol.replace('/', '')
            
            response = self.session.get_positions(
                category="spot",
                symbol=pybit_symbol
            )
            
            if response['retCode'] != 0:
                print(f"⚠️  Position check error: {response['retMsg']}")
                return None
                
            positions = response['result']['list']
            for pos in positions:
                if float(pos['size']) > 0:
                    return {
                        'side': pos['side'],
                        'size': float(pos['size']),
                        'avg_price': float(pos['avgPrice']),
                        'pnl': float(pos['unrealisedPnl']),
                        'pnl_pct': float(pos['unrealisedPnl']) / float(pos['positionValue']) * 100 if float(pos['positionValue']) > 0 else 0
                    }
            return None
            
        except Exception as e:
            print(f"⚠️  Position check error: {e}")
            return None
    
    async def get_balance(self):
        """Get USDT balance"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if response['retCode'] != 0:
                print(f"⚠️  Balance error: {response['retMsg']}")
                return 0
                
            for coin in response['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    return float(coin['availableToWithdraw'])
            return 0
            
        except Exception as e:
            print(f"⚠️  Balance error: {e}")
            return 0
    
    async def place_market_order(self, symbol, side, qty):
        """Place market order using pybit"""
        try:
            pybit_symbol = symbol.replace('/', '')
            
            response = self.session.place_order(
                category="spot",
                symbol=pybit_symbol,
                side=side.title(),  # Buy or Sell
                orderType="Market",
                qty=str(qty)
            )
            
            if response['retCode'] == 0:
                return {
                    'success': True,
                    'orderId': response['result']['orderId'],
                    'orderLinkId': response['result']['orderLinkId']
                }
            else:
                print(f"❌ Order failed: {response['retMsg']}")
                return {'success': False, 'error': response['retMsg']}
                
        except Exception as e:
            print(f"❌ Order error: {e}")
            return {'success': False, 'error': str(e)}
    
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
                        'size': 0.1
                    }
                    self.trade_count += 1
                    await self.telegram.trade_opened(symbol, signal['price'], 0.1)
                    
                elif signal['action'] == 'SELL' and symbol in self.positions:
                    if self.positions[symbol]['side'] == 'long':
                        entry = self.positions[symbol]['entry_price']
                        pnl_pct = ((signal['price'] - entry) / entry) * 100
                        pnl_usd = pnl_pct * 10
                        
                        print(f"[DEMO] Position closed - P&L: {pnl_pct:+.2f}%")
                        await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                        
                        if pnl_pct > 0:
                            self.win_count += 1
                        else:
                            self.loss_count += 1
                            
                        del self.positions[symbol]
                        
                print(f"{'='*50}\n")
                return
            
            # Real trading
            balance = await self.get_balance()
            
            if balance < 10:
                print("❌ Insufficient USDT balance for trading")
                return
            
            current_position = await self.get_current_position(symbol)
            
            if signal['action'] == 'BUY' and not current_position:
                # Calculate position size
                position_value = self.risk_manager.calculate_position_size(balance, signal['price'])
                
                # Place buy order
                result = await self.place_market_order(symbol, 'buy', position_value)
                
                if result['success']:
                    print(f"✅ BUY order executed")
                    print(f"Order ID: {result['orderId']}")
                    print(f"Amount: {position_value:.4f} {symbol.split('/')[0]}")
                    
                    self.trade_count += 1
                    await self.telegram.trade_opened(symbol, signal['price'], position_value)
                else:
                    print(f"❌ Buy order failed: {result['error']}")
                    
            elif signal['action'] == 'SELL' and current_position:
                # Close position
                result = await self.place_market_order(symbol, 'sell', current_position['size'])
                
                if result['success']:
                    pnl_pct = current_position.get('pnl_pct', 0)
                    pnl_usd = current_position.get('pnl', 0)
                    
                    print(f"✅ Position closed")
                    print(f"Order ID: {result['orderId']}")
                    print(f"P&L: {pnl_pct:+.2f}% (${pnl_usd:.2f})")
                    
                    await self.telegram.trade_closed(symbol, pnl_pct, pnl_usd)
                    
                    if pnl_pct > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                else:
                    print(f"❌ Sell order failed: {result['error']}")
                    
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
        print(f"Exchange: Bybit (pybit)")
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
                    df = await self.fetch_ohlcv(symbol, timeframe='1', limit=100)
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