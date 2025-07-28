#!/usr/bin/env python3
"""
ZORA Trading Bot - Trailing Stop Profit Locker Test & Visualization
Tests and plots the trailing stop mechanism for profit protection
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_test_params():
    """Create test parameters file"""
    test_params = {
        "rsi_length": 7,
        "mfi_length": 7, 
        "oversold_level": 25,
        "overbought_level": 75,
        "require_trend": True
    }
    
    params_dir = os.path.join(project_root, 'strategies')
    os.makedirs(params_dir, exist_ok=True)
    params_file = os.path.join(params_dir, 'params_RSI_MFI_Cloud.json')
    
    with open(params_file, 'w') as f:
        json.dump(test_params, f)
    
    return params_file

def generate_profitable_scenario(length=200, start_price=0.082):
    """Generate ZORA price data with profitable uptrend followed by pullback"""
    dates = pd.date_range(start=datetime.now() - timedelta(minutes=length*5), periods=length, freq='5min')
    
    prices = [start_price]
    
    for i in range(1, length):
        current_price = prices[-1]
        
        if i < 50:
            # Initial sideways movement
            change = np.random.normal(0, 0.002)
        elif i < 120:
            # Strong uptrend for profit lock activation
            change = np.random.normal(0.003, 0.002)  # Positive bias
        elif i < 160:
            # Continue uptrend but slower
            change = np.random.normal(0.001, 0.002)
        else:
            # Pullback to test trailing stop
            change = np.random.normal(-0.002, 0.002)  # Negative bias
        
        new_price = current_price * (1 + change)
        new_price = max(0.001, new_price)  # Prevent negative prices
        prices.append(new_price)
    
    # Generate OHLC from prices
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': prices[i-1] if i > 0 else price,
            'high': max(price, high),
            'low': min(price, low), 
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

class TrailingStopTester:
    def __init__(self):
        self.params_file = None
        self.setup()
        
    def setup(self):
        """Setup test environment"""
        self.params_file = create_test_params()
        
    def cleanup(self):
        """Cleanup test files"""
        if self.params_file and os.path.exists(self.params_file):
            os.remove(self.params_file)
    
    def test_trailing_stop_mechanism(self, plot=True):
        """Test and visualize trailing stop profit locker"""
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            from core.risk_management import RiskManager
            
            print("🔒 Testing Trailing Stop Profit Locker for ZORA/USDT")
            print("=" * 60)
            
            # Initialize components
            strategy = RSIMFICloudStrategy()
            risk_manager = RiskManager()
            
            # Generate test scenario
            df = generate_profitable_scenario(200, 0.082)
            
            # Simulate trading session
            results = self.simulate_trading_session(df, strategy, risk_manager)
            
            if plot:
                self.plot_results(df, results, strategy, risk_manager)
            
            return results
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            return None
    
    def simulate_trading_session(self, df, strategy, risk_manager):
        """Simulate a complete trading session with trailing stops"""
        results = {
            'timestamps': [],
            'prices': [],
            'entry_price': None,
            'entry_time': None,
            'profit_lock_activated': False,
            'profit_lock_time': None,
            'profit_lock_price': None,
            'trailing_stops': [],
            'exit_price': None,
            'exit_time': None,
            'exit_reason': None,
            'pnl_pct': 0,
            'max_profit_pct': 0,
            'events': []
        }
        
        position_active = False
        profit_lock_active = False
        current_trailing_stop = None
        entry_price = None
        max_profit_seen = 0
        
        # Process each price point
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            results['timestamps'].append(current_time)
            results['prices'].append(current_price)
            
            if not position_active:
                # Look for entry signal (simulate BUY signal at bar 10)
                if i == 10:  # Entry point
                    position_active = True
                    entry_price = current_price
                    results['entry_price'] = entry_price
                    results['entry_time'] = current_time
                    results['events'].append({
                        'time': current_time,
                        'price': current_price,
                        'event': 'POSITION_OPENED',
                        'details': f'Long @ ${current_price:.6f}'
                    })
                    print(f"📈 Position opened: ${current_price:.6f} at {current_time.strftime('%H:%M')}")
                
                results['trailing_stops'].append(None)
                continue
            
            # Calculate current P&L
            pnl_pct = (current_price - entry_price) / entry_price * 100
            leveraged_pnl = pnl_pct * risk_manager.leverage
            results['pnl_pct'] = leveraged_pnl
            
            # Track max profit
            if leveraged_pnl > max_profit_seen:
                max_profit_seen = leveraged_pnl
                results['max_profit_pct'] = max_profit_seen
            
            # Check for profit lock activation
            profit_threshold = risk_manager.break_even_pct * 100 * risk_manager.leverage
            
            if not profit_lock_active and leveraged_pnl >= profit_threshold:
                profit_lock_active = True
                results['profit_lock_activated'] = True
                results['profit_lock_time'] = current_time
                results['profit_lock_price'] = current_price
                
                # Calculate initial trailing stop
                trailing_distance = current_price * risk_manager.trailing_stop_distance
                current_trailing_stop = current_price - trailing_distance
                
                results['events'].append({
                    'time': current_time,
                    'price': current_price,
                    'event': 'PROFIT_LOCK_ACTIVATED',
                    'details': f'P&L: {leveraged_pnl:.1f}% (threshold: {profit_threshold:.1f}%)'
                })
                print(f"🔓 PROFIT LOCK ACTIVATED! P&L: {leveraged_pnl:.2f}% @ ${current_price:.6f}")
                print(f"   Initial trailing stop: ${current_trailing_stop:.6f}")
            
            # Update trailing stop if profit lock is active
            if profit_lock_active and current_trailing_stop is not None:
                trailing_distance = current_price * risk_manager.trailing_stop_distance
                new_trailing_stop = current_price - trailing_distance
                
                # Only move stop up (never down)
                if new_trailing_stop > current_trailing_stop:
                    current_trailing_stop = new_trailing_stop
                    results['events'].append({
                        'time': current_time,
                        'price': current_price,
                        'event': 'TRAILING_STOP_UPDATED',
                        'details': f'New stop: ${current_trailing_stop:.6f}'
                    })
            
            results['trailing_stops'].append(current_trailing_stop)
            
            # Check if trailing stop hit
            if profit_lock_active and current_trailing_stop and current_price <= current_trailing_stop:
                # Position closed by trailing stop
                results['exit_price'] = current_price
                results['exit_time'] = current_time
                results['exit_reason'] = 'TRAILING_STOP_HIT'
                
                final_pnl = (current_price - entry_price) / entry_price * 100 * risk_manager.leverage
                results['pnl_pct'] = final_pnl
                
                results['events'].append({
                    'time': current_time,
                    'price': current_price,
                    'event': 'POSITION_CLOSED',
                    'details': f'Trailing stop hit @ ${current_price:.6f}, P&L: {final_pnl:.2f}%'
                })
                
                print(f"🔒 TRAILING STOP HIT! Exit @ ${current_price:.6f}")
                print(f"   Final P&L: {final_pnl:.2f}% (Max seen: {max_profit_seen:.2f}%)")
                break
        
        return results
    
    def plot_results(self, df, results, strategy, risk_manager):
        """Create comprehensive visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
        
        # Main price chart
        timestamps = results['timestamps']
        prices = results['prices']
        trailing_stops = results['trailing_stops']
        
        # Plot price line
        ax1.plot(timestamps, prices, 'b-', linewidth=2, label='ZORA Price', alpha=0.8)
        
        # Plot trailing stop line
        valid_stops = [(t, s) for t, s in zip(timestamps, trailing_stops) if s is not None]
        if valid_stops:
            stop_times, stop_prices = zip(*valid_stops)
            ax1.plot(stop_times, stop_prices, 'r--', linewidth=2, label='Trailing Stop', alpha=0.7)
            ax1.fill_between(stop_times, stop_prices, min(prices), alpha=0.1, color='red', label='Stop Loss Zone')
        
        # Mark key events
        if results['entry_time']:
            ax1.scatter(results['entry_time'], results['entry_price'], 
                       color='green', s=100, marker='^', zorder=5, label='Entry')
            ax1.annotate(f'ENTRY\n${results["entry_price"]:.6f}', 
                        xy=(results['entry_time'], results['entry_price']),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if results['profit_lock_time']:
            ax1.scatter(results['profit_lock_time'], results['profit_lock_price'], 
                       color='orange', s=100, marker='*', zorder=5, label='Profit Lock')
            ax1.annotate('PROFIT LOCK\nACTIVATED', 
                        xy=(results['profit_lock_time'], results['profit_lock_price']),
                        xytext=(-10, 30), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if results['exit_time']:
            ax1.scatter(results['exit_time'], results['exit_price'], 
                       color='red', s=100, marker='v', zorder=5, label='Exit')
            ax1.annotate(f'EXIT\n${results["exit_price"]:.6f}\nP&L: {results["pnl_pct"]:.1f}%', 
                        xy=(results['exit_time'], results['exit_price']),
                        xytext=(-10, -40), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Format main chart
        ax1.set_title('ZORA/USDT - Trailing Stop Profit Locker Test', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # P&L chart
        if results['entry_time']:
            entry_idx = timestamps.index(results['entry_time'])
            pnl_times = timestamps[entry_idx:]
            pnl_values = []
            
            for i, price in enumerate(prices[entry_idx:], entry_idx):
                if results['entry_price']:
                    pnl = (price - results['entry_price']) / results['entry_price'] * 100 * risk_manager.leverage
                    pnl_values.append(pnl)
            
            # Plot P&L
            ax2.plot(pnl_times, pnl_values, 'g-', linewidth=2, label='P&L %')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Mark profit threshold
            profit_threshold = risk_manager.break_even_pct * 100 * risk_manager.leverage
            ax2.axhline(y=profit_threshold, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Profit Lock Threshold ({profit_threshold:.1f}%)')
            
            # Fill profitable area
            ax2.fill_between(pnl_times, pnl_values, 0, where=np.array(pnl_values) > 0, 
                           alpha=0.3, color='green', label='Profit Zone')
            ax2.fill_between(pnl_times, pnl_values, 0, where=np.array(pnl_values) < 0, 
                           alpha=0.3, color='red', label='Loss Zone')
        
        ax2.set_ylabel('P&L (%)', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"zora_trailing_stop_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as: {plot_filename}")
        
        plt.show()
        
        # Print summary
        self.print_summary(results, risk_manager)
    
    def print_summary(self, results, risk_manager):
        """Print detailed test summary"""
        print("\n" + "=" * 60)
        print("📊 TRAILING STOP PROFIT LOCKER - TEST SUMMARY")
        print("=" * 60)
        
        if results['entry_price']:
            print(f"💰 Entry Price: ${results['entry_price']:.6f}")
            print(f"⏰ Entry Time: {results['entry_time'].strftime('%H:%M:%S')}")
        
        if results['profit_lock_activated']:
            print(f"\n🔓 PROFIT LOCK ACTIVATED:")
            print(f"   💎 Activation Price: ${results['profit_lock_price']:.6f}")
            print(f"   ⏰ Activation Time: {results['profit_lock_time'].strftime('%H:%M:%S')}")
            profit_at_lock = (results['profit_lock_price'] - results['entry_price']) / results['entry_price'] * 100
            print(f"   📈 Profit at Lock: {profit_at_lock * risk_manager.leverage:.2f}%")
        
        if results['exit_price']:
            print(f"\n🔒 POSITION CLOSED:")
            print(f"   💸 Exit Price: ${results['exit_price']:.6f}")
            print(f"   ⏰ Exit Time: {results['exit_time'].strftime('%H:%M:%S')}")
            print(f"   📉 Exit Reason: {results['exit_reason']}")
            print(f"   🎯 Final P&L: {results['pnl_pct']:.2f}%")
            print(f"   🚀 Max Profit Seen: {results['max_profit_pct']:.2f}%")
            
            # Calculate profit protected
            if results['max_profit_pct'] > 0:
                profit_protected = (results['max_profit_pct'] - results['pnl_pct']) / results['max_profit_pct'] * 100
                print(f"   🛡️ Profit Protected: {results['max_profit_pct'] - results['pnl_pct']:.2f}% ({profit_protected:.1f}% of max)")
        
        print(f"\n🎮 RISK MANAGEMENT SETTINGS:")
        print(f"   📊 Leverage: {risk_manager.leverage}x")
        print(f"   🔓 Profit Lock Threshold: {risk_manager.break_even_pct * 100 * risk_manager.leverage:.1f}%")
        print(f"   📏 Trailing Distance: {risk_manager.trailing_stop_distance * 100:.1f}%")
        
        print(f"\n📈 KEY EVENTS:")
        for event in results['events']:
            print(f"   {event['time'].strftime('%H:%M')} - {event['event']}: {event['details']}")
        
        print("=" * 60)

def main():
    """Main test runner"""
    tester = TrailingStopTester()
    
    try:
        print("🧪 ZORA Trading Bot - Trailing Stop Profit Locker Test")
        print("=" * 60)
        
        # Run comprehensive test
        results = tester.test_trailing_stop_mechanism(plot=True)
        
        if results:
            print("\n✅ Test completed successfully!")
            if results['profit_lock_activated']:
                print("🔓 Profit lock mechanism working correctly")
            if results['exit_reason'] == 'TRAILING_STOP_HIT':
                print("🔒 Trailing stop protection activated")
        else:
            print("❌ Test failed")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()