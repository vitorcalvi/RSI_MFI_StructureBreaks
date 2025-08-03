import os
import asyncio
import pandas as pd
import numpy as np
import json
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
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.position_start_time = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Initialize tracking dictionaries
        profit_target = self.risk_manager.config['fixed_break_even_threshold']
        self.exit_reasons = {f'profit_target_${profit_target}': 0, 'emergency_stop': 0, 'profit_lock': 0}
        self.rejections = {'extreme_rsi': 0, 'extreme_mfi': 0, 'zero_volume': 0, 'counter_trend': 0, 'low_confidence': 0, 'total_signals': 0}
        
        self._set_symbol_rules()
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
    
    def _set_symbol_rules(self):
        """Set symbol-specific trading rules"""
        rules = {'ETH': ('0.01', 0.01), 'BTC': ('0.001', 0.001), 'ADA': ('1', 1.0)}
        for key, (step, min_qty) in rules.items():
            if key in self.symbol:
                self.qty_step, self.min_qty = step, min_qty
                return
        self.qty_step, self.min_qty = '1', 1.0
    
    def connect(self):
        """Connect to exchange"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_quantity(self, qty):
        """Format quantity according to exchange rules"""
        if qty < self.min_qty:
            return "0"
        try:
            decimals = len(self.qty_step.split('.')[1]) if '.' in self.qty_step else 0
            qty_step_float = float(self.qty_step)
            rounded_qty = round(qty / qty_step_float) * qty_step_float
            return f"{rounded_qty:.{decimals}f}" if decimals > 0 else str(int(rounded_qty))
        except:
            return f"{qty:.3f}"
    
    def _get_market_data(self):
        """Get market indicators and momentum using strategy's trend detection"""
        if len(self.price_data) < 20:
            return {'rsi': 50, 'mfi': 50, 'ema3': 0, 'ema7': 0, 'ema15': 0, 'volume_ratio': 1, 'trend': 'neutral', 'strength': 0, 'direction': '→'}
        
        close = self.price_data['close']
        volume = self.price_data['volume']
        
        # Get indicators from strategy
        indicators = self.strategy.calculate_indicators(self.price_data)
        rsi = max(0, min(100, indicators.get('rsi', pd.Series([50])).iloc[-1])) if 'rsi' in indicators else 50
        mfi = max(0, min(100, indicators.get('mfi', pd.Series([50])).iloc[-1])) if 'mfi' in indicators else 50
        
        # Calculate EMAs and get trend from strategy
        ema3, ema7, ema15 = close.ewm(span=3).mean().iloc[-1], close.ewm(span=7).mean().iloc[-1], close.ewm(span=15).mean().iloc[-1]
        trend = self.strategy.detect_trend(self.price_data)
        
        # Volume and strength calculations
        vol_avg = volume.tail(20).mean()
        volume_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 and volume.iloc[-1] > 0 else 0
        momentum = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) > 2 else 0
        
        if trend == 'strong_uptrend':
            direction, strength = '↗', min(100, max(abs(momentum) * 5000, 50))
        elif trend == 'strong_downtrend':
            direction, strength = '↘', min(100, max(abs(momentum) * 5000, 50))
        else:
            direction, strength = '→', min(40, max(abs(momentum) * 2500, 10))
        
        return {'rsi': rsi, 'mfi': mfi, 'ema3': ema3, 'ema7': ema7, 'ema15': ema15, 'volume_ratio': volume_ratio, 'trend': trend, 'strength': strength, 'direction': direction}
    
    def _log_trade(self, action, price, **kwargs):
        """Trade logging with market context"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        market_data = self._get_market_data()
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'side': signal.get('action', ''), 'price': round(price, 2), 'size': kwargs.get('quantity', ''),
                'rsi': round(signal.get('rsi', 0), 1), 'mfi': round(signal.get('mfi', 0), 1),
                'trend': signal.get('trend', 'neutral'), 'confidence': round(signal.get('confidence', 0), 1),
                'volatility': round(market_data.get('volatility', 0), 3), 'ema3': round(market_data['ema3'], 2),
                'volume_ratio': round(market_data['volume_ratio'], 2)
            }
        else:
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 2), 'pnl': round(kwargs.get('pnl', 0), 2),
                'rsi_exit': round(market_data['rsi'], 1),
                'mfi_exit': round(market_data['mfi'], 1)
            }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass

    async def run_cycle(self):
        """Run one trading cycle"""
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        elif not self.position:
            signal = self.strategy.generate_signal(self.price_data)
            if signal:
                self.rejections['total_signals'] += 1
                await self._execute_trade(signal)
        
        self._display_status()
    
    async def _update_market_data(self):
        """Update market data"""
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').set_index('timestamp')
            return True
        except:
            return False
    
    async def _check_position_status(self):
        """Check position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    await self._on_position_closed()
                self._reset_position()
            else:
                if not self.position:
                    self.position_start_time = datetime.now()
                self.position = pos_list[0]
        except:
            pass
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
    
    async def _check_position_exit(self):
        """Check position exit conditions"""
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        
        should_close, reason = self.risk_manager.should_close_position(current_price, entry_price, side, unrealized_pnl)
        if should_close:
            await self._close_position(reason)
    
    async def _execute_trade(self, signal):
        """Execute trade with validation"""
        current_price = float(self.price_data['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or not self._validate_signal(signal, self._get_market_data())[0]:
            return
        
        if not self.risk_manager.validate_trade(signal, balance, current_price)[0]:
            return
        
        qty = self.risk_manager.calculate_position_size(balance, current_price, signal['structure_stop'])
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            return
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Market", qty=formatted_qty, timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty)
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, self.strategy.get_strategy_info())
        except:
            pass
    
    def _validate_signal(self, signal, market_data):
        """Signal validation using strategy config"""
        rsi, mfi = signal.get('rsi', 50), signal.get('mfi', 50)
        side, confidence = signal.get('action', ''), signal.get('confidence', 0)
        strategy_config = self.strategy.config
        
        # Check extreme values
        if rsi < 5 or rsi > 95:
            self.rejections['extreme_rsi'] += 1
            return False, f"Extreme RSI {rsi:.1f}"
        
        if mfi < 5 or mfi > 95:
            self.rejections['extreme_mfi'] += 1
            return False, f"Extreme MFI {mfi:.1f}"
        
        # Check volume and confidence
        if market_data['volume_ratio'] == 0:
            self.rejections['zero_volume'] += 1
            return False, "Zero volume detected"
        
        if confidence < 70:
            self.rejections['low_confidence'] += 1
            return False, f"Low confidence {confidence:.1f}"
        
        # Strategy-specific validation
        if (side == 'SELL' and rsi < strategy_config['short_rsi_minimum']) or (side == 'BUY' and rsi > strategy_config['uptrend_oversold']):
            self.rejections['counter_trend'] += 1
            return False, f"RSI {rsi:.1f} invalid for {side.lower()}"
        
        return True, "Valid"
    
    async def _close_position(self, reason="Manual"):
        """Close position"""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
        pnl = float(self.position.get('unrealisedPnl', 0))
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = self.format_quantity(float(self.position['size']))
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Market", qty=qty, timeInForce="IOC", reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                self._track_exit_reason(reason)
                self._log_trade("EXIT", current_price, reason=reason, pnl=pnl)
                
                exit_data = {'trigger': reason, 'rsi': self._get_market_data()['rsi'], 'mfi': self._get_market_data()['mfi']}
                await self.notifier.send_trade_exit(exit_data, current_price, pnl, 0, self.strategy.get_strategy_info())
        except:
            pass
    
    def _track_exit_reason(self, reason):
        """Track exit reasons"""
        profit_key = f'profit_target_${self.risk_manager.config["fixed_break_even_threshold"]}'
        
        if 'profit_target' in reason or 'profit_lock' in reason:
            self.exit_reasons[profit_key] += 1
        elif reason in self.exit_reasons:
            self.exit_reasons[reason] += 1
        else:
            self.exit_reasons.setdefault('manual_exit', 0)
            self.exit_reasons['manual_exit'] += 1
    
    async def get_account_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
        except:
            pass
        return 0
    
    async def _on_position_closed(self):
        """Handle externally closed position"""
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", pnl=pnl)

    def analyze_recent_trades(self, limit=10):
        """Quick analysis of recent trades"""
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
            
            exits = [json.loads(line.strip()) for line in lines[-limit*2:] if 'EXIT' in line]
            if not exits:
                return
            
            wins = len([t for t in exits if t.get('pnl', 0) > 0])
            avg_pnl = sum(t.get('pnl', 0) for t in exits) / len(exits)
            
            print(f"\n📊 Last {len(exits)} trades: {wins}W/{len(exits)-wins}L | Avg PnL: ${avg_pnl:.2f}")
        except:
            pass

    def _display_status(self):
        """Display enhanced status using actual config values"""
        try:
            price = float(self.price_data['close'].iloc[-1])
            time = self.price_data.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            market_data = self._get_market_data()
            
            print("\n" * 50)
            
            # Header
            w = 77
            print(f"{'='*w}\n⚡  {symbol_display} HIGH-FREQUENCY SCALPING BOT\n{'='*w}\n")
            
            # Strategy setup - use actual config values
            strategy_config = self.strategy.config
            risk_config = self.risk_manager.config
            
            print("⚙️  STRATEGY SETUP\n" + "─"*w)
            print(f"📊 RSI/MFI High-Frequency Scalping Strategy")
            print(f"📈 RSI({strategy_config['rsi_length']}) MFI({strategy_config['mfi_length']}) │ 🔥 Cooldown: {strategy_config['cooldown_seconds']}s")
            print(f"💰 Thresholds: ≤{strategy_config['uptrend_oversold']} Uptrend │ ≥{strategy_config['downtrend_overbought']} Downtrend")
            print("─"*w + "\n")

            # Risk Management
            print("🛡️  RISK MANAGEMENT\n" + "─"*w)
            profit_target = risk_config['fixed_break_even_threshold']
            print(f"💵 Position Size: ${risk_config['fixed_position_usdt']:,} USDT │ 🎯 Profit Target: ${profit_target} │ ⚡ Leverage: {risk_config['leverage']}x")
            print(f"📊 Reward Ratio: {risk_config['reward_ratio']}:1")
            print("─"*w + "\n")

            # Market momentum
            print("📈  MARKET MOMENTUM\n" + "─"*w)
            trend_display = f"{market_data['trend'].replace('_', ' ').title()}"
            direction_emoji = {"↗": "🟢", "↘": "🔴", "→": "🟡", "↕": "🟠"}.get(market_data['direction'], "🟡")
            print(f"🎯 Trend: {trend_display:<12} │ 💪 Strength: {market_data['strength']:>3.0f}% │ {market_data['direction']} Direction")
            print(f"📊 EMA3: {direction_emoji} │ EMA7: {direction_emoji} │ EMA15: {direction_emoji} = {trend_display} {market_data['direction']}")
            print("─"*w + "\n")

            # Exit reasons and rejections
            print("📊  EXIT REASONS & SIGNAL FILTERS\n" + "─"*w)
            profit_key = f'profit_target_${profit_target}'
            
            print(f"🎯 {profit_key:<17} : {self.exit_reasons.get(profit_key, 0):2d} │ 🚨 emergency_stop : {self.exit_reasons.get('emergency_stop', 0):2d} │ 💰 profit_lock : {self.exit_reasons.get('profit_lock', 0):2d}")
            print(f"⏰ max_hold_time     : {self.exit_reasons.get('max_hold_time', 0):2d}")
            
            if self.rejections.get('total_signals', 0) > 0:
                print(f"🚫 Signals rejected  : {self.rejections.get('extreme_rsi', 0):2d} RSI │ {self.rejections.get('extreme_mfi', 0):2d} MFI │ {self.rejections.get('zero_volume', 0):2d} Vol │ {self.rejections.get('counter_trend', 0):2d} Trend")
                total_signals = self.rejections['total_signals']
                acceptance_rate = (self.trade_id / total_signals * 100) if total_signals > 0 else 0
                print(f"📈 Signal rate       : {self.trade_id}/{total_signals} accepted ({acceptance_rate:.1f}%)")
            
            print("─"*w + "\n")

            # Current status
            print(f"⏰ {time}   |   💰 ${price_formatted}")
            print(f"📈 RSI: {market_data['rsi']:.1f}  |   MFI: {market_data['mfi']:.1f}  |   {trend_display.upper()}{market_data['direction']}")
            print()
            
            # Position info
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry = float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side} Position: {size} @ ${entry:.2f}")
                print(f"   PnL: ${pnl:.2f} | Age: {age:.1f}s")
            else:
                print("⚡  No Position — scanning…")
            
            # Show quick trade analysis
            self.analyze_recent_trades(7)
            print("─" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _print_strategy_section(self, strategy_info, strategy_config, w):
        """Print strategy setup section"""
        print("⚙️  STRATEGY SETUP\n" + "─"*w)
        print(f"📊 {strategy_info['name']}")
        print(f"📈 RSI({strategy_config['rsi_length']}) MFI({strategy_config['mfi_length']}) │ 🔥 Cooldown: {strategy_config['cooldown_seconds']}s")
        print(f"💰 Thresholds: ≤{strategy_config['uptrend_oversold']} Uptrend │ ≥{strategy_config['downtrend_overbought']} Downtrend")
        print("─"*w + "\n")
    
    def _print_risk_section(self, risk_config, w):
        """Print risk management section"""
        print("🛡️  RISK MANAGEMENT\n" + "─"*w)
        print(f"💵 Position Size: ${risk_config['fixed_position_usdt']:,} USDT │ 🎯 Profit Target: ${risk_config['fixed_break_even_threshold']} │ ⚡ Leverage: {risk_config['leverage']}x")
        print(f"🚨 Emergency Stop: ${risk_config['emergency_stop_amount']:,} │ 📊 Reward Ratio: {risk_config['reward_ratio']}:1")
        print("─"*w + "\n")
    
    def _print_market_section(self, market_data, w):
        """Print market momentum section"""
        print("📈  MARKET MOMENTUM\n" + "─"*w)
        
        trend_display = {'strong_uptrend': 'STRONG UP', 'strong_downtrend': 'STRONG DOWN', 'neutral': 'NEUTRAL'}.get(market_data['trend'], market_data['trend'].upper())
        print(f"🎯 Trend: {trend_display:<10} │ 💪 Strength: {market_data['strength']:>3.0f}% │ {market_data['direction']} Direction")
        
        # EMA indicators
        current_price = float(self.price_data['close'].iloc[-1])
        ema_indicators = ["🟢" if current_price > market_data[f'ema{n}'] else "🔴" for n in [3, 7, 15]]
        
        pattern_labels = {
            "🟢🟢🟢": "Bullish 📈", "🔴🔴🔴": "Bearish 📉", "🟢🟢🔴": "Weak Bull 📈", "🔴🔴🟢": "Weak Bear 📉",
            "🟢🔴🟢": "Mixed ↕️", "🔴🟢🔴": "Mixed ↕️", "🟢🔴🔴": "Very Weak 📈", "🔴🟢🟢": "Very Weak 📉"
        }
        pattern_label = pattern_labels.get("".join(ema_indicators), "Choppy")
        
        print(f"📊 EMA3: {ema_indicators[0]} │ EMA7: {ema_indicators[1]} │ EMA15: {ema_indicators[2]} = {pattern_label}")
        print("─"*w + "\n")
    
    def _print_stats_section(self, risk_config, w):
        """Print exit reasons and signal filters section"""
        print("📊  EXIT REASONS & SIGNAL FILTERS\n" + "─"*w)
        profit_key = f'profit_target_${risk_config["fixed_break_even_threshold"]}'
        
        print(f"🎯 {profit_key:<17} : {self.exit_reasons[profit_key]:2d} │ 🚨 emergency_stop : {self.exit_reasons['emergency_stop']:2d} │ 💰 profit_lock : {self.exit_reasons['profit_lock']:2d}")
        
        if self.rejections['total_signals'] > 0:
            print(f"🚫 Signals rejected  : {self.rejections['extreme_rsi']:2d} RSI │ {self.rejections['extreme_mfi']:2d} MFI │ {self.rejections['zero_volume']:2d} Vol │ {self.rejections['counter_trend']:2d} Trend")
            print(f"📈 Signal rate       : {self.trade_id}/{self.rejections['total_signals']} accepted ({(self.trade_id/self.rejections['total_signals']*100):.1f}%)")
        
        print("─"*w + "\n")
    
    def _print_status_section(self, time, price_formatted, market_data, risk_config):
        """Print current status and position section"""
        trend_status = {'strong_uptrend': 'STRONG↗', 'strong_downtrend': 'STRONG↘', 'neutral': 'NEUTRAL'}.get(market_data['trend'], market_data['trend'])
        
        print(f"⏰ {time}   |   💰 ${price_formatted}")
        print(f"📈 RSI: {market_data['rsi']:.1f}  |   MFI: {market_data['mfi']:.1f}  |   {trend_status}")
        print()
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            size = self.position.get('size', '0')
            side = self.position.get('side', '')
            
            pnl_pct = (pnl / (float(size) * entry)) * 100 if entry > 0 and size != '0' else 0
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side} Position: {size} @ ${entry:.2f}")
            print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        else:
            print("⚡  No Position — scanning…")
        
        self.analyze_recent_trades(5)
        print("─" * 60)