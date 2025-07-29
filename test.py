#!/usr/bin/env python3
"""
FIXED REVERSAL THRESHOLD TEST
Tests the ACTUAL reversal logic with correct calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class FixedRiskManager:
    def __init__(self):
        self.leverage = 25
        self.max_position_size = 0.002  # 0.2% as decimal
        self.stop_loss_pct = 0.015      # 1.5%
        
        # Reversal thresholds (based on wallet P&L)
        self.profit_lock_threshold = 0.5        # 0.5% wallet P&L
        self.profit_protection_threshold = 2.0  # 2.0% wallet P&L  
        self.loss_reversal_threshold = -1.0     # -1.0% wallet P&L
        
    def should_reverse_for_loss(self, wallet_pnl_pct):
        return wallet_pnl_pct <= self.loss_reversal_threshold
    
    def should_activate_profit_lock(self, wallet_pnl_pct):
        return wallet_pnl_pct >= self.profit_lock_threshold
    
    def should_take_profit_protection(self, wallet_pnl_pct):
        return wallet_pnl_pct >= self.profit_protection_threshold

class FixedReversalTester:
    def __init__(self):
        self.risk_manager = FixedRiskManager()
        self.results_dir = "fixed_reversal_plots"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def calculate_wallet_pnl(self, entry_price, current_price, balance=10000):
        """Calculate wallet P&L correctly"""
        # Position size in tokens
        position_size_tokens = self.risk_manager.max_position_size * balance / entry_price
        
        # Price difference
        price_diff = current_price - entry_price
        
        # Unrealized P&L in USD
        unrealized_pnl = price_diff * position_size_tokens
        
        # Wallet P&L percentage (THIS IS THE KEY)
        wallet_pnl_pct = (unrealized_pnl / balance) * 100
        
        return wallet_pnl_pct, unrealized_pnl, position_size_tokens
    
    def test_realistic_reversal_scenario(self):
        """Test with realistic price movements that should trigger reversal"""
        print("🔥 TESTING REALISTIC REVERSAL SCENARIOS")
        print("=" * 60)
        
        entry_price = 0.085
        balance = 10000
        
        # Calculate what price change is needed for -1% wallet P&L
        position_size_tokens = self.risk_manager.max_position_size * balance / entry_price
        print(f"📊 Position Details:")
        print(f"   • Entry Price: ${entry_price:.6f}")
        print(f"   • Balance: ${balance:,.0f}")
        print(f"   • Position Size: {self.risk_manager.max_position_size*100:.3f}% = {position_size_tokens:.0f} tokens")
        print(f"   • Position Value: ${position_size_tokens * entry_price:.2f}")
        
        # Calculate price needed for -1% wallet P&L
        # -1% of $10,000 = -$100
        # Price drop needed = $100 / position_size_tokens
        loss_needed = balance * 0.01  # $100 for -1%
        price_drop_needed = loss_needed / position_size_tokens
        reversal_price = entry_price - price_drop_needed
        price_change_pct = (price_drop_needed / entry_price) * 100
        
        print(f"\n🎯 Reversal Trigger Calculation:")
        print(f"   • Loss needed for -1% wallet P&L: ${loss_needed:.2f}")
        print(f"   • Price drop needed: ${price_drop_needed:.6f}")
        print(f"   • Reversal price: ${reversal_price:.6f}")
        print(f"   • Price change needed: {price_change_pct:.2f}%")
        
        # Create test scenarios
        scenarios = {
            'Just Before Reversal': entry_price - (price_drop_needed * 0.9),  # 90% of needed drop
            'Exactly at Reversal': reversal_price,                           # Exact reversal point
            'Past Reversal': entry_price - (price_drop_needed * 1.2),       # 120% of needed drop
            'Profit Zone': entry_price * 1.03,                              # +3% price = profit
        }
        
        # Test each scenario
        results = []
        for name, test_price in scenarios.items():
            wallet_pnl_pct, unrealized_pnl, tokens = self.calculate_wallet_pnl(entry_price, test_price, balance)
            price_change = ((test_price - entry_price) / entry_price) * 100
            
            # Test triggers
            should_reverse = self.risk_manager.should_reverse_for_loss(wallet_pnl_pct)
            profit_lock = self.risk_manager.should_activate_profit_lock(wallet_pnl_pct)
            profit_protection = self.risk_manager.should_take_profit_protection(wallet_pnl_pct)
            
            results.append({
                'scenario': name,
                'test_price': test_price,
                'price_change_pct': price_change,
                'wallet_pnl_pct': wallet_pnl_pct,
                'unrealized_pnl': unrealized_pnl,
                'should_reverse': should_reverse,
                'profit_lock': profit_lock,
                'profit_protection': profit_protection
            })
        
        # Display results
        print(f"\n📋 REVERSAL TEST RESULTS:")
        print("=" * 80)
        print(f"{'Scenario':<20} {'Price':<12} {'Price Δ':<10} {'Wallet P&L':<12} {'Reverse':<8} {'Action':<15}")
        print("-" * 80)
        
        for r in results:
            action = "REVERSE" if r['should_reverse'] else \
                    "PROTECT" if r['profit_protection'] else \
                    "LOCK" if r['profit_lock'] else "HOLD"
            
            print(f"{r['scenario']:<20} ${r['test_price']:<11.6f} {r['price_change_pct']:>+7.2f}% "
                  f"{r['wallet_pnl_pct']:>+9.2f}% {'YES' if r['should_reverse'] else 'NO':<7} {action:<15}")
        
        # Create visualization
        self.create_reversal_visualization(results, entry_price, balance)
        
        return results
    
    def create_reversal_visualization(self, results, entry_price, balance):
        """Create comprehensive reversal visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        scenarios = [r['scenario'] for r in results]
        prices = [r['test_price'] for r in results]
        price_changes = [r['price_change_pct'] for r in results]
        wallet_pnls = [r['wallet_pnl_pct'] for r in results]
        reversals = [r['should_reverse'] for r in results]
        
        # Colors based on action
        colors = []
        for r in results:
            if r['should_reverse']:
                colors.append('red')
            elif r['profit_protection']:
                colors.append('darkgreen')
            elif r['profit_lock']:
                colors.append('orange')
            else:
                colors.append('gray')
        
        # Plot 1: Price levels
        bars1 = ax1.bar(range(len(scenarios)), prices, color=colors, alpha=0.7)
        ax1.axhline(y=entry_price, color='blue', linestyle='--', linewidth=2, label='Entry Price')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Test Scenarios - Price Levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add price values on bars
        for i, (bar, price) in enumerate(zip(bars1, prices)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.00005,
                    f'${price:.6f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        # Plot 2: Price changes
        bars2 = ax2.bar(range(len(scenarios)), price_changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.set_ylabel('Price Change (%)')
        ax2.set_title('Price Changes from Entry')
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, change) in enumerate(zip(bars2, price_changes)):
            height = bar.get_height()
            y_offset = 0.1 if height >= 0 else -0.3
            ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{change:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Plot 3: Wallet P&L with thresholds
        bars3 = ax3.bar(range(len(scenarios)), wallet_pnls, color=colors, alpha=0.7)
        
        # Add threshold lines
        ax3.axhline(y=self.risk_manager.profit_protection_threshold, color='darkgreen', 
                   linestyle='--', linewidth=2, label='Profit Protection (2%)')
        ax3.axhline(y=self.risk_manager.profit_lock_threshold, color='orange', 
                   linestyle='--', linewidth=2, label='Profit Lock (0.5%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=self.risk_manager.loss_reversal_threshold, color='red', 
                   linestyle='--', linewidth=2, label='Loss Reversal (-1%)')
        
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
        ax3.set_ylabel('Wallet P&L (%)')
        ax3.set_title('Wallet P&L vs Thresholds')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, pnl) in enumerate(zip(bars3, wallet_pnls)):
            height = bar.get_height()
            y_offset = 0.05 if height >= 0 else -0.15
            ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{pnl:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # Plot 4: Action summary
        actions = []
        action_colors = []
        for r in results:
            if r['should_reverse']:
                actions.append('REVERSE')
                action_colors.append('red')
            elif r['profit_protection']:
                actions.append('PROTECT')
                action_colors.append('darkgreen')
            elif r['profit_lock']:
                actions.append('LOCK')
                action_colors.append('orange')
            else:
                actions.append('HOLD')
                action_colors.append('gray')
        
        # Create action chart
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_names = list(action_counts.keys())
        action_values = list(action_counts.values())
        action_chart_colors = ['red' if a == 'REVERSE' else 
                              'darkgreen' if a == 'PROTECT' else 
                              'orange' if a == 'LOCK' else 'gray' for a in action_names]
        
        pie = ax4.pie(action_values, labels=action_names, colors=action_chart_colors, 
                     autopct='%1.0f%%', startangle=90)
        ax4.set_title('Actions Triggered in Test Scenarios')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/fixed_reversal_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_price_movement_series(self):
        """Test with continuous price movement to see reversal trigger"""
        print(f"\n🔄 TESTING CONTINUOUS PRICE MOVEMENT")
        print("=" * 60)
        
        entry_price = 0.085
        balance = 10000
        
        # Create price series from entry to significant loss
        # Need about -5% price move to get -1% wallet P&L
        price_range = np.linspace(entry_price, entry_price * 0.94, 100)  # -6% price range
        
        results = []
        reversal_triggered = False
        reversal_point = None
        
        for i, price in enumerate(price_range):
            wallet_pnl_pct, unrealized_pnl, tokens = self.calculate_wallet_pnl(entry_price, price, balance)
            should_reverse = self.risk_manager.should_reverse_for_loss(wallet_pnl_pct)
            
            if should_reverse and not reversal_triggered:
                reversal_triggered = True
                reversal_point = i
                print(f"🔴 REVERSAL TRIGGERED at step {i}:")
                print(f"   • Price: ${price:.6f}")
                print(f"   • Price change: {((price - entry_price) / entry_price) * 100:.2f}%")
                print(f"   • Wallet P&L: {wallet_pnl_pct:.2f}%")
            
            results.append({
                'step': i,
                'price': price,
                'price_change_pct': ((price - entry_price) / entry_price) * 100,
                'wallet_pnl_pct': wallet_pnl_pct,
                'unrealized_pnl': unrealized_pnl,
                'should_reverse': should_reverse
            })
        
        # Create plot
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price movement
        ax1.plot(df['step'], df['price'], 'b-', linewidth=2, label='ZORA Price')
        ax1.axhline(y=entry_price, color='green', linestyle='--', alpha=0.7, label='Entry Price')
        
        if reversal_point is not None:
            ax1.scatter(reversal_point, df.iloc[reversal_point]['price'], 
                       color='red', s=200, marker='v', zorder=5, label='Reversal Triggered')
            ax1.axvline(x=reversal_point, color='red', linestyle=':', alpha=0.5)
        
        ax1.set_title('Continuous Price Movement Test')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wallet P&L
        ax2.plot(df['step'], df['wallet_pnl_pct'], 'g-', linewidth=2, label='Wallet P&L %')
        ax2.axhline(y=self.risk_manager.loss_reversal_threshold, color='red', 
                   linestyle='--', linewidth=2, label='Reversal Threshold (-1%)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Shade reversal zone
        ax2.axhspan(self.risk_manager.loss_reversal_threshold, -5, alpha=0.2, color='red', 
                   label='Reversal Zone')
        
        if reversal_point is not None:
            ax2.scatter(reversal_point, df.iloc[reversal_point]['wallet_pnl_pct'], 
                       color='red', s=200, marker='v', zorder=5)
            ax2.axvline(x=reversal_point, color='red', linestyle=':', alpha=0.5)
        
        ax2.set_title('Wallet P&L Progression')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Wallet P&L (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/continuous_movement_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df, reversal_point
    
    def run_comprehensive_test(self):
        """Run all fixed tests"""
        print("🔧 FIXED REVERSAL THRESHOLD TEST")
        print("=" * 70)
        print("Testing with CORRECT calculations")
        print()
        
        # Test realistic scenarios
        scenario_results = self.test_realistic_reversal_scenario()
        
        # Test continuous movement
        movement_results, reversal_point = self.test_price_movement_series()
        
        print(f"\n" + "=" * 70)
        print("✅ FIXED REVERSAL TESTING COMPLETE!")
        print("=" * 70)
        
        # Summary
        reversal_scenarios = [r for r in scenario_results if r['should_reverse']]
        print(f"📊 SUMMARY:")
        print(f"   • Scenarios tested: {len(scenario_results)}")
        print(f"   • Reversals triggered: {len(reversal_scenarios)}")
        print(f"   • Continuous test reversal at step: {reversal_point if reversal_point else 'None'}")
        
        if reversal_scenarios:
            print(f"\n🔴 REVERSAL TRIGGERS:")
            for r in reversal_scenarios:
                print(f"   • {r['scenario']}: {r['price_change_pct']:+.2f}% price = {r['wallet_pnl_pct']:+.2f}% wallet P&L")
        
        print(f"\n📁 Plots saved in: {self.results_dir}/")
        print("=" * 70)

if __name__ == "__main__":
    tester = FixedReversalTester()
    tester.run_comprehensive_test()