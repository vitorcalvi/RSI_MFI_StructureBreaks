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
        
        self._init_exchange_connection()
        self._init_trading_state()
        self._init_tracking_data()
        self._set_symbol_rules()
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
    
    def _init_exchange_connection(self):
        """Initialize exchange connection parameters."""
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
    
    def _init_trading_state(self):
        """Initialize trading state variables."""
        self.position = None
        self.position_start_time = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
    
    def _init_tracking_data(self):
        """Initialize tracking dictionaries."""
        profit_target = self.risk_manager.config['fixed_break_even_threshold']
        self.exit_reasons = {
            f'profit_target_${profit_target}': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'profit_lock': 0, 'trailing_stop': 0, 'position_closed': 0,
            'bot_shutdown': 0, 'manual_exit': 0
        }
        
        self.rejections = {
            'extreme_rsi': 0, 'extreme_mfi': 0, 'zero_volume': 0,
            'counter_trend': 0, 'low_confidence': 0, 'total_signals': 0
        }
    
    def _set_symbol_rules(self):
        """Set symbol-specific trading rules."""
        rules = {'ETH': ('0.01', 0.01), 'BTC': ('0.001', 0.001), 'ADA': ('1', 1.0)}
        for key, (step, min_qty) in rules.items():
            if key in self.symbol:
                self.qty_step, self.min_qty = step, min_qty
                return
        self.qty_step, self.min_qty = '1', 1.0
    
    def connect(self):
        """Connect to exchange with error handling."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception:
            return False
    
    def format_quantity(self, qty):
        """Format quantity according to exchange rules."""
        if qty < self.min_qty:
            return "0"
        try:
            decimals = len(self.qty_step.split('.')[1]) if '.' in self.qty_step else 0
            qty_step_float = float(self.qty_step)
            rounded_qty = round(qty / qty_step_float) * qty_step_float
            return f"{rounded_qty:.{decimals}f}" if decimals > 0 else str(int(rounded_qty))
        except Exception:
            return f"{qty:.3f}"
    
    def _calculate_market_indicators(self):
        """Calculate market indicators and momentum."""
        if len(self.price_data) < 20:
            return self._get_default_market_data()
        
        close = self.price_data['close']
        volume = self.price_data['volume']
        
        # Get indicators with validation
        indicators = self.strategy.calculate_indicators(self.price_data)
        rsi = self._safe_get_indicator(indicators.get('rsi'), 50)
        mfi = self._safe_get_indicator(indicators.get('mfi'), 50)
        
        # Calculate momentum and volatility
        returns = close.pct_change().tail(10)
        volatility = returns.std() if len(returns) > 1 else 0
        momentum_5m = self._safe_momentum_calc(close, 5)
        momentum_20m = self._safe_momentum_calc(close, 20)
        
        # Volume analysis
        vol_avg = volume.tail(20).mean()
        current_vol = volume.iloc[-1]
        volume_ratio = current_vol / vol_avg if vol_avg > 0 and current_vol > 0 else 1
        
        # Reject trades with zero volume
        if current_vol == 0 or vol_avg == 0:
            volume_ratio = 0
        
        # Trend determination
        trend, direction, strength = self._determine_trend(momentum_5m, momentum_20m)
        volume_strength = self._calculate_volume_strength(current_vol, vol_avg)
        
        return {
            'rsi': rsi, 'mfi': mfi, 'volatility': volatility,
            'momentum_5m': momentum_5m, 'momentum_20m': momentum_20m, 'volume_ratio': volume_ratio,
            'trend': trend, 'strength': strength, 'direction': direction, 'volume_strength': volume_strength
        }
    
    def _get_default_market_data(self):
        """Get default market data when insufficient data."""
        return {'rsi': 50, 'mfi': 50, 'volatility': 0, 'momentum_5m': 0, 'momentum_20m': 0, 
               'volume_ratio': 1, 'trend': 'NEUTRAL', 'strength': 0, 'direction': '→', 'volume_strength': 0}
    
    def _safe_get_indicator(self, indicator, default):
        """Safely get indicator value with bounds checking."""
        if indicator is None or len(indicator) == 0:
            return default
        value = indicator.iloc[-1] if hasattr(indicator, 'iloc') else indicator
        return max(0, min(100, value)) if pd.notna(value) else default
    
    def _safe_momentum_calc(self, close, periods):
        """Safely calculate momentum with bounds checking."""
        if len(close) <= periods:
            return 0
        return ((close.iloc[-1] - close.iloc[-periods-1]) / close.iloc[-periods-1]) * 100
    
    def _determine_trend(self, momentum_5m, momentum_20m):
        """Determine market trend based on momentum."""
        if abs(momentum_5m) > 2 or abs(momentum_20m) > 5:
            strength = min(100, max(abs(momentum_5m) * 20, abs(momentum_20m) * 10))
            if momentum_5m > 0.5 and momentum_20m > 0:
                return 'BULLISH', '↗', strength
            elif momentum_5m < -0.5 and momentum_20m < 0:
                return 'BEARISH', '↘', strength
            else:
                return 'MIXED', '↕', strength
        return 'NEUTRAL', '→', 0
    
    def _calculate_volume_strength(self, current_vol, vol_avg):
        """Calculate volume strength."""
        if vol_avg <= 0:
            return 0
        vol_momentum = ((current_vol - vol_avg) / vol_avg) * 100
        return min(100, max(0, vol_momentum))
    
    def _log_trade(self, action, price, **kwargs):
        """Log trade with market context."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        market_data = self._calculate_market_indicators()
        
        if action == "ENTRY":
            self.trade_id += 1
            log_data = self._create_entry_log(timestamp, price, market_data, kwargs)
        else:
            log_data = self._create_exit_log(timestamp, price, market_data, kwargs)
        
        self._write_log(log_data)
    
    def _create_entry_log(self, timestamp, price, market_data, kwargs):
        """Create entry log data."""
        signal = kwargs.get('signal', {})
        return {
            'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
            'side': signal.get('action', ''), 'price': round(price, 2), 'size': kwargs.get('quantity', ''),
            'rsi': round(signal.get('rsi', 0), 1), 'mfi': round(signal.get('mfi', 0), 1),
            'trend': signal.get('trend', 'neutral'), 'confidence': round(signal.get('confidence', 0), 1),
            'volatility': round(market_data['volatility'], 3), 'momentum': round(market_data['momentum_5m'], 2),
            'volume_ratio': round(market_data['volume_ratio'], 2)
        }
    
    def _create_exit_log(self, timestamp, price, market_data, kwargs):
        """Create exit log data."""
        duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
        return {
            'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
            'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
            'price': round(price, 2), 'pnl': round(kwargs.get('pnl', 0), 2),
            'hold_seconds': round(duration, 1), 'rsi_exit': round(market_data['rsi'], 1),
            'mfi_exit': round(market_data['mfi'], 1)
        }
    
    def _write_log(self, log_data):
        """Write log data to file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception:
            pass

    async def run_cycle(self):
        """Run one trading cycle."""
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._process_new_signals()
        
        self._display_status()
    
    async def _process_new_signals(self):
        """Process new trading signals."""
        signal = self.strategy.generate_signal(self.price_data)
        if signal:
            self.rejections['total_signals'] += 1
            await self._execute_trade(signal)
    
    async def _update_market_data(self):
        """Update market data from exchange."""
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
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
        except Exception:
            return False
    
    async def _check_position_status(self):
        """Check current position status."""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    await self._on_position_closed()
                self._reset_position()
                return
            
            if not self.position:
                self.position_start_time = datetime.now()
            self.position = pos_list[0]
        except Exception:
            pass
    
    def _reset_position(self):
        """Reset position state."""
        self.position = None
        self.position_start_time = None
    
    async def _check_position_exit(self):
        """Check if position should be closed."""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age
        )
        
        if should_close:
            await self._close_position(reason)
    
    async def _execute_trade(self, signal):
        """Execute trade with validation."""
        current_price = float(self.price_data['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or not self._validate_signal(signal, current_price):
            return
        
        qty = self.risk_manager.calculate_position_size(balance, current_price, signal['structure_stop'])
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            return
        
        await self._place_order(signal, formatted_qty, current_price)
    
    def _validate_signal(self, signal, current_price):
        """Validate signal with enhanced checks."""
        market_data = self._calculate_market_indicators()
        
        # Enhanced validation using strategy config
        rsi, mfi = signal.get('rsi', 50), signal.get('mfi', 50)
        
        validation_checks = [
            (5 <= rsi <= 95, f"Extreme RSI {rsi:.1f}", 'extreme_rsi'),
            (5 <= mfi <= 95, f"Extreme MFI {mfi:.1f}", 'extreme_mfi'),
            (market_data['volume_ratio'] > 0, "Zero volume detected", 'zero_volume'),
            (signal.get('confidence', 0) >= 70, f"Low confidence {signal.get('confidence', 0):.1f}", 'low_confidence')
        ]
        
        for condition, error_msg, rejection_key in validation_checks:
            if not condition:
                self.rejections[rejection_key] += 1
                print(f"❌ Trade rejected: {error_msg}")
                return False
        
        # Original risk manager validation
        balance = 10000  # Placeholder for actual balance
        is_valid, _ = self.risk_manager.validate_trade(signal, balance, current_price)
        return is_valid
    
    async def _place_order(self, signal, formatted_qty, current_price):
        """Place order on exchange."""
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Market", qty=formatted_qty, timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty)
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, 
                                                   self.strategy.get_strategy_info())
        except Exception:
            pass
    
    async def _close_position(self, reason="Manual"):
        """Close current position."""
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
                await self._process_exit(current_price, pnl, reason)
        except Exception:
            pass
    
    async def _process_exit(self, current_price, pnl, reason):
        """Process position exit."""
        duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
        market_data = self._calculate_market_indicators()
        
        self._track_exit_reason(reason)
        self._log_trade("EXIT", current_price, reason=reason, pnl=pnl)
        
        exit_data = {'trigger': reason, 'rsi': market_data['rsi'], 'mfi': market_data['mfi']}
        await self.notifier.send_trade_exit(exit_data, current_price, pnl, duration, 
                                          self.strategy.get_strategy_info())
    
    def _track_exit_reason(self, reason):
        """Track exit reason statistics."""
        profit_target = self.risk_manager.config['fixed_break_even_threshold']
        profit_key = f'profit_target_${profit_target}'
        
        reason_mapping = {
            'profit_target': profit_key, 'profit_lock': profit_key,
            'max_hold_time_exceeded': 'max_hold_time',
            'Bot shutdown': 'bot_shutdown', 'Manual': 'manual_exit'
        }
        
        mapped_reason = reason_mapping.get(reason, reason)
        if mapped_reason in self.exit_reasons:
            self.exit_reasons[mapped_reason] += 1
        else:
            self.exit_reasons['manual_exit'] += 1
    
    async def get_account_balance(self):
        """Get account balance from exchange."""
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
            return 0
        except Exception:
            return 0
    
    async def _on_position_closed(self):
        """Handle externally closed position."""
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", pnl=pnl)

    def analyze_recent_trades(self, limit=10):
        """Analyze recent trades performance."""
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
            
            trades = []
            for line in lines[-limit*2:]:
                try:
                    trades.append(json.loads(line.strip()))
                except Exception:
                    continue
            
            exits = [t for t in trades if t['action'] == 'EXIT']
            if not exits:
                return
            
            wins = len([t for t in exits if t['pnl'] > 0])
            avg_pnl = sum(t['pnl'] for t in exits) / len(exits)
            avg_hold = sum(t['hold_seconds'] for t in exits) / len(exits)
            
            print(f"\n📊 Last {len(exits)} trades: {wins}W/{len(exits)-wins}L | "
                  f"Avg PnL: ${avg_pnl:.2f} | Hold: {avg_hold:.1f}s")
        except Exception:
            pass

    def _display_status(self):
        """Display trading status with simplified formatting."""
        try:
            price = float(self.price_data['close'].iloc[-1])
            time_str = self.price_data.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            market_data = self._calculate_market_indicators()
            
            self._print_header(symbol_display)
            self._print_strategy_setup()
            self._print_market_momentum(market_data)
            self._print_exit_reasons_and_rejections()
            self._print_current_status(time_str, price_formatted, market_data)
            self._print_position_info()
            self.analyze_recent_trades(5)
            print("─" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _print_header(self, symbol_display):
        """Print status header."""
        print("\n" * 50)
        w = 77
        print(f"{'='*w}\n⚡  {symbol_display} HIGH-FREQUENCY SCALPING BOT\n{'='*w}\n")
    
    def _print_strategy_setup(self):
        """Print strategy configuration."""
        strategy_config = self.strategy.config
        risk_config = self.risk_manager.config
        w = 77
        
        print("⚙️  STRATEGY SETUP\n" + "─"*w)
        print(f"📊 RSI({strategy_config['rsi_length']}) MFI({strategy_config['mfi_length']}) │ "
              f"🔥 Cooldown: {strategy_config['cooldown_seconds']}s  │ ⚡ Mode: FIXED-SIZE")
        print(f"💰 Position Size: ${risk_config['fixed_position_usdt']:,} USDT │ "
              f"📈 Uptrend: ≤{strategy_config['uptrend_oversold']}  │ "
              f"📉 Downtrend: ≥{strategy_config['downtrend_overbought']}")
        print("─"*w + "\n")
    
    def _print_market_momentum(self, market_data):
        """Print market momentum information."""
        w = 77
        print("📈  MARKET MOMENTUM\n" + "─"*w)
        print(f"🎯 Trend: {market_data['trend']:<8} │ 💪 Strength: {market_data['strength']:>3.0f}% │ "
              f"{market_data['direction']} Direction")
        print(f"⚡ 5min: {market_data['momentum_5m']:>+5.2f}% │ 📊 20min: {market_data['momentum_20m']:>+5.2f}% │ "
              f"📈 Volume: {market_data['volume_strength']:>3.0f}%")
        print("─"*w + "\n")
    
    def _print_exit_reasons_and_rejections(self):
        """Print exit reasons and signal rejections."""
        w = 77
        risk_config = self.risk_manager.config
        profit_target = risk_config['fixed_break_even_threshold']
        profit_key = f'profit_target_${profit_target}'
        
        print("📊  EXIT REASONS & SIGNAL FILTERS\n" + "─"*w)
        print(f"🎯 {profit_key:<17} : {self.exit_reasons[profit_key]:2d} │ "
              f"🚨 emergency_stop : {self.exit_reasons['emergency_stop']:2d} │ "
              f"⏰ max_hold_time   : {self.exit_reasons['max_hold_time']:2d}")
        print(f"💰 profit_lock       : {self.exit_reasons['profit_lock']:2d} │ "
              f"📉 trailing_stop  : {self.exit_reasons['trailing_stop']:2d} │ "
              f"🔄 position_closed : {self.exit_reasons['position_closed']:2d}")
        
        if self.rejections['total_signals'] > 0:
            print(f"🚫 Signals rejected  : {self.rejections['extreme_rsi']:2d} RSI │ "
                  f"{self.rejections['extreme_mfi']:2d} MFI │ {self.rejections['zero_volume']:2d} Vol │ "
                  f"{self.rejections['counter_trend']:2d} Trend")
            acceptance_rate = (self.trade_id/self.rejections['total_signals']*100)
            print(f"📈 Signal rate       : {self.trade_id}/{self.rejections['total_signals']} "
                  f"accepted ({acceptance_rate:.1f}%)")
        
        print("─"*w + "\n")
    
    def _print_current_status(self, time_str, price_formatted, market_data):
        """Print current market status."""
        print(f"⏰ {time_str}   |   💰 ${price_formatted}")
        print(f"📈 RSI: {market_data['rsi']:.1f}  |   MFI: {market_data['mfi']:.1f}")
        print()
    
    def _print_position_info(self):
        """Print current position information."""
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            size = self.position.get('size', '0')
            side = self.position.get('side', '')
            
            pnl_pct = (pnl / (float(size) * entry)) * 100 if entry > 0 and size != '0' else 0
            age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
            max_hold = self.risk_manager.config['max_position_time']
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side} Position: {size} @ ${entry:.2f}")
            print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Age: {age:.1f}s / {max_hold}s")
        else:
            print("⚡  No Position — scanning…")