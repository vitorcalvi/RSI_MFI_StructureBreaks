#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

class AutoTester:
    def __init__(self, data_path, custom_fees=None):
        self.data_path = data_path
        self.custom_fees = custom_fees
        self.results = []
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_backtest(self, params):
        # Override fees if custom fees specified
        if self.custom_fees is not None:
            params = params.copy()
            params['entry_fee_pct'] = self.custom_fees
            params['exit_fee_pct'] = self.custom_fees
        
        bt = EthHFTBacktester(self.data_path, params)
        return bt.run_backtest()
    
    def test_all_configs(self):
        config_dir = Path('strategies/bot_param_scenarios_1000')
        config_files = list(config_dir.glob('*.json'))
        
        fee_msg = f" (using custom fees: {self.custom_fees*100:.3f}%)" if self.custom_fees else ""
        print(f"Testing {len(config_files)} configurations from {config_dir}{fee_msg}...")
        
        for config_file in config_files:
            try:
                config = self.load_config(config_file)
                result = self.run_backtest(config)
                
                self.results.append({
                    'config': config_file.name,
                    'return_pct': result['return_pct'],
                    'total_trades': result['total_trades'],
                    'trades_per_minute': result['trades_per_minute'],
                    'status': 'SUCCESS'
                })
                print(f"✅ {config_file.name}: {result['return_pct']:.2f}% | {result['total_trades']} trades | {result['trades_per_minute']:.3f}/min")
                
            except Exception as e:
                self.results.append({
                    'config': config_file.name,
                    'return_pct': 0.0,
                    'total_trades': 0,
                    'trades_per_minute': 0.0,
                    'status': f'ERROR: {str(e)}'
                })
                print(f"❌ {config_file.name}: FAILED - {e}")
    
    def generate_report(self):
        df = pd.DataFrame(self.results)
        
        # Clean up NaN and infinite values
        df['return_pct'] = df['return_pct'].replace([np.inf, -np.inf], 0.0)
        df['return_pct'] = df['return_pct'].fillna(0.0)
        
        df = df.sort_values('return_pct', ascending=False)
        
        print("\n" + "="*80)
        print("AUTOMATED TEST RESULTS")
        print("="*80)
        print(f"{'STATUS':<2} {'CONFIG':<25} {'RETURN':<8} {'TRADES':<7} {'PER MIN':<8}")
        print("-"*80)
        
        for _, row in df.iterrows():
            status = "✅" if row['status'] == 'SUCCESS' else "❌"
            return_str = f"{row['return_pct']:>7.2f}%" if np.isfinite(row['return_pct']) else "  ERROR"
            trades_str = f"{row['total_trades']:>6}" if 'total_trades' in row else "    0"
            freq_str = f"{row['trades_per_minute']:>7.3f}" if 'trades_per_minute' in row else "   0.000"
            print(f"{status} {row['config']:<25} {return_str} {trades_str} {freq_str}")
        
        print("="*80)
        successful = df[df['status'] == 'SUCCESS']
        if len(successful) > 0:
            best = successful.iloc[0]
            print(f"🏆 BEST CONFIG: {best['config']}")
            print(f"   Return: {best['return_pct']:.2f}% | Trades: {best['total_trades']} | Frequency: {best['trades_per_minute']:.3f}/min")
        
        # Top 5 by profitability
        print("\n" + "="*50)
        print("🏆 TOP 5 MOST PROFITABLE")
        print("="*50)
        top_profit = successful.head(5)
        for i, (_, row) in enumerate(top_profit.iterrows(), 1):
            print(f"{i}. {row['config']:<25} {row['return_pct']:>7.2f}%")
        
        # Top 5 by trade volume
        print("\n" + "="*50) 
        print("📊 TOP 5 MOST ACTIVE (Highest Trades)")
        print("="*50)
        top_trades = successful.nlargest(5, 'total_trades')
        for i, (_, row) in enumerate(top_trades.iterrows(), 1):
            freq_per_hour = row['trades_per_minute'] * 60
            print(f"{i}. {row['config']:<25} {row['total_trades']:>4} trades ({freq_per_hour:.1f}/hr)")
        
        print("="*80)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'test_results_{timestamp}.csv', index=False)
        print(f"📊 Results saved to: test_results_{timestamp}.csv")

class EthHFTBacktester:
    def __init__(self, data_path, params):
        self.data_path = data_path
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.trades = []
        
        # Load params
        self.rsi_length = params['rsi_length']
        self.mfi_length = params['mfi_length']
        self.oversold = params['oversold']
        self.overbought = params['overbought']
        self.structure_lookback = params['structure_lookback']
        self.structure_buffer_pct = params['structure_buffer_pct']
        self.fixed_risk_pct = params['fixed_risk_pct']
        self.trailing_stop_pct = params['trailing_stop_pct']
        self.profit_lock_threshold = params['profit_lock_threshold']
        self.reward_ratio = params['reward_ratio']
        self.entry_fee_pct = params['entry_fee_pct']
        self.exit_fee_pct = params['exit_fee_pct']
        
    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'], index_col='timestamp')
        self.data = df.resample('1min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
    
    def compute_indicators(self):
        close = self.data['close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(alpha=1/self.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.data['rsi'] = 100 - 100/(1+rs)
        
        tp = (self.data['high'] + self.data['low'] + close) / 3
        mf = tp * self.data['volume']
        sign = tp.diff()
        pos_mf = mf.where(sign > 0, 0)
        neg_mf = mf.where(sign <= 0, 0)
        
        pos_ema = pos_mf.ewm(alpha=1/self.mfi_length, adjust=False).mean()
        neg_ema = neg_mf.ewm(alpha=1/self.mfi_length, adjust=False).mean()
        mf_ratio = pos_ema / neg_ema.replace(0, np.nan)
        self.data['mfi'] = 100 - 100/(1+mf_ratio)
        
        self.data[['rsi', 'mfi']] = self.data[['rsi', 'mfi']].fillna(50)
    
    def run_backtest(self):
        self.load_data()
        self.compute_indicators()
        self.trades.clear()
        self.balance = self.initial_balance
        position = None
        
        for ts, row in self.data.iterrows():
            price = row['close']
            
            if position is None:
                if row['rsi'] < self.oversold and row['mfi'] < self.oversold:
                    position = self.open_position('BUY', price, ts)
                elif row['rsi'] > self.overbought and row['mfi'] > self.overbought:
                    position = self.open_position('SELL', price, ts)
            else:
                if self.manage_position(position, price, ts):
                    position = None
        
        if position:
            self.close_position(position, self.data['close'].iloc[-1], self.data.index[-1])
        
        return self.evaluate()
    
    def open_position(self, side, price, ts):
        risk_amount = self.balance * self.fixed_risk_pct
        window = self.data.loc[:ts].iloc[-self.structure_lookback:]
        
        if side == 'BUY':
            stop_level = window['low'].min() - price * (self.structure_buffer_pct / 100)
        else:
            stop_level = window['high'].max() + price * (self.structure_buffer_pct / 100)
        
        # Prevent division by zero
        stop_distance = abs(price - stop_level)
        if stop_distance == 0 or stop_distance < price * 0.0001:  # Min 0.01% distance
            return None
        
        size = risk_amount / stop_distance
        
        # Validate size
        if size <= 0 or not np.isfinite(size):
            return None
        
        if side == 'BUY':
            tp = price + (price - stop_level) * self.reward_ratio
        else:
            tp = price - (stop_level - price) * self.reward_ratio
        
        entry_cost = price * size * self.entry_fee_pct
        
        # Validate entry cost
        if not np.isfinite(entry_cost) or entry_cost >= self.balance:
            return None
            
        self.balance -= entry_cost
        
        return {
            'side': side, 'entry': price, 'stop': stop_level,
            'tp': tp, 'size': size, 'open_ts': ts
        }
    
    def manage_position(self, pos, price, ts):
        pnl_pct = ((price - pos['entry']) / pos['entry'] * 100) if pos['side'] == 'BUY' else ((pos['entry'] - price) / pos['entry'] * 100)
        
        # Stop loss
        if (pos['side'] == 'BUY' and price <= pos['stop']) or (pos['side'] == 'SELL' and price >= pos['stop']):
            self.close_position(pos, price, ts)
            return True
        
        # Take profit
        if (pos['side'] == 'BUY' and price >= pos['tp']) or (pos['side'] == 'SELL' and price <= pos['tp']):
            self.close_position(pos, price, ts)
            return True
        
        # Trailing stop
        if pnl_pct >= self.profit_lock_threshold:
            new_stop = price * (1 - self.trailing_stop_pct) if pos['side'] == 'BUY' else price * (1 + self.trailing_stop_pct)
            pos['stop'] = max(pos['stop'], new_stop) if pos['side'] == 'BUY' else min(pos['stop'], new_stop)
        
        return False
    
    def close_position(self, pos, price, ts):
        size = pos['size']
        
        # Validate inputs
        if not np.isfinite(size) or size <= 0:
            return
            
        pnl = (price - pos['entry']) * size if pos['side'] == 'BUY' else (pos['entry'] - price) * size
        exit_cost = price * size * self.exit_fee_pct
        
        # Validate calculations
        if not np.isfinite(pnl) or not np.isfinite(exit_cost):
            return
            
        net_pnl = pnl - exit_cost
        self.balance += net_pnl
        
        # Calculate duration in minutes
        duration_minutes = (ts - pos['open_ts']).total_seconds() / 60
        
        self.trades.append({
            'pnl': net_pnl,
            'open_ts': pos['open_ts'],
            'close_ts': ts,
            'duration_minutes': duration_minutes
        })
    
    def evaluate(self):
        if not self.trades:
            return {'return_pct': 0.0, 'total_trades': 0, 'trades_per_minute': 0.0}
        
        df = pd.DataFrame(self.trades)
        total_pnl = df['pnl'].sum()
        
        # Handle NaN/infinite results
        if not np.isfinite(total_pnl):
            return {'return_pct': 0.0, 'total_trades': len(df), 'trades_per_minute': 0.0}
        
        # Calculate trading frequency per minute
        time_span_minutes = (self.data.index[-1] - self.data.index[0]).total_seconds() / 60
        trades_per_minute = len(df) / time_span_minutes if time_span_minutes > 0 else 0
        
        return {
            'return_pct': total_pnl / self.initial_balance * 100,
            'total_trades': len(df),
            'trades_per_minute': trades_per_minute
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Config Tester for Trading Strategies')
    parser.add_argument('--data', '-d', default='_data/ETHUSDT_1_7d_20250731_071705.csv',
                       help='Path to data file (default: _data/ETHUSDT_1_7d_20250731_071705.csv)')
    parser.add_argument('--fee', '-f', type=float, default=None,
                       help='Override fees for all configs (e.g., --fee 0.0055 for 0.55%%)')
    
    args = parser.parse_args()
    
    # Check if config directory exists
    if not os.path.exists('strategies/bot_param_scenarios'):
        print("❌ Config directory not found: strategies/bot_param_scenarios")
        exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("Use --data to specify correct path")
        exit(1)
    
    tester = AutoTester(args.data, args.fee)
    tester.test_all_configs()
    tester.generate_report()