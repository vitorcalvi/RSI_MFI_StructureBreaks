#!/usr/bin/env python3
"""
CORRECTED REVERSAL THRESHOLD TEST
Fixed the position size calculation error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class CorrectedRiskManager:
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

class CorrectedReversalTester:
    def __init__(self):
        self.risk_manager = CorrectedRiskManager()
        self.results_dir = "corrected_reversal_plots"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def calculate_wallet_pnl_correct(self, entry_price, current_price, balance=10000):
        """CORRECTED: Calculate wallet P&L with proper position sizing"""
        
        # CORRECT position sizing: 0.2% of balance in USD value
        position_value_usd = balance * self.risk_manager.max_position_size  # $20 for $10k balance
        
        # Convert USD position value to tokens at entry price
        position_size_tokens = position_value_usd / entry_price
        
        # Price difference per token
        price_diff_per_token = current_price - entry_price
        
        # Total unrealized P&L in USD
        unrealized_pnl_usd = price_diff_per_token * position_size_tokens
        
        # Wallet P&L percentage 
        wallet_pnl_pct = (unrealized_pnl_usd / balance) * 100
        
        return wallet_pnl_pct, unrealized_pnl_usd, position_size_tokens, position_value_usd
    
    def test_realistic_reversal_scenario(self):
        """Test with CORRECTED calculations"""
        print("🔧 TESTING WITH CORRECTED CALCULATIONS")
        print("=" * 60)
        
        entry_price = 0.085
        balance = 10000
        
        # Calculate position details CORRECTLY
        position_value_usd = balance * self.risk_manager.max_position_size
        position_size_tokens = position_value_usd / entry_price
        
        print(f"📊 CORRECTED Position Details:")
        print(f"   • Entry Price: ${entry_price:.6f}")
        print(f"   • Balance: ${balance:,.0f}")
        print(f"   • Position Size: {self.risk_manager.max_position_size*100:.3f}% = ${position_value_usd:.2f}")
        print(f"   • Tokens: {position_size_tokens:.0f} tokens")
        
        # Calculate what price change is needed for -1% wallet P&L
        # For -1% wallet P&L: need -$100 loss on $10,000 balance
        loss_needed_usd = balance * 0.01  # $100
        
        # Price drop needed per token to achieve $100 total loss
        price_drop_per_token = loss_needed_usd / position_size_tokens
        
        # Reversal trigger price
        reversal_price = entry_price - price_drop_per_token
        price_change_pct = (price_drop_per_token / entry_price) * 100
        
        print(f"\n🎯 CORRECTED Reversal Trigger Calculation:")
        print(f"   • Loss needed for -1% wallet P&L: ${loss_needed_usd:.2f}")
        print(f"   • Price drop needed per token: ${price_drop_per_token:.6f}")
        print(f"   • Reversal price: ${reversal_price:.6f}")
        print(f"   • Price change needed: {price_change_pct:.2f}%")
        
        # Create realistic test scenarios
        scenarios = {
            'Small Loss (-0.5%)': entry_price - (price_drop_per_token * 0.5),   # Half the needed drop
            'Almost Reversal (-0.9%)': entry_price - (price_drop_per_token * 0.9), # 90% of needed drop
            'Exact Reversal (-1.0%)': reversal_price,                           # Exact reversal point
            'Past Reversal (-1.5%)': entry_price - (price_drop_per_token * 1.5), # 150% of needed drop
            'Profit Lock (+0.5%)': entry_price * 1.025,                        # +2.5% price for profit lock
            'Profit Protection (+2%)': entry_price * 1.10,                     # +10% price for protection
        }
        
        # Test each scenario
        results = []
        for name, test_price in scenarios.items():
            wallet_pnl_pct, unrealized_pnl, tokens, pos_value = self.calculate_wallet_pnl_correct(
                entry_price, test_price, balance)
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
        print(f"\n📋 CORRECTED REVERSAL TEST RESULTS:")
        print("=" * 90)
        print(f"{'Scenario':<25} {'Price':<12} {'Price Δ':<10} {'Wallet P&L':<12} {'USD P&L':<10} {'Action':<15}")
        print("-" * 90)
        
        for r in results:
            action = "PROTECT" if r['profit_protection'] else \
                    "LOCK" if r['profit_lock'] else \
                    "REVERSE" if r['should_reverse'] else "HOLD"
            
            print(f"{r['scenario']:<25} ${r['test_price']:<11.6f} {r['price_change_pct']:>+7.2f}% "
                  f"{r['wallet_pnl_pct']:>+9.2f}% {r['unrealized_pnl']:>+8.2f} {action:<15}")
        
        # Create visualization
        self.create_corrected_visualization(results, entry_price, balance)
        
        return results
    
    def create_corrected_visualization(self, results, entry_price, balance):
        """Create corrected visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        scenarios = [r['scenario'] for r in results]
        prices = [r['test_price'] for r in results]
        price_changes = [r['price_change_pct'] for r in results]
        wallet_pnls = [r['wallet_pnl_pct'] for r in results]
        
        # Colors based on action
        colors = []
        for r in results:
            if r['profit_protection']:
                colors.append('darkgreen')
            elif r['profit_lock']:
                colors.append('orange')
            elif r['should_reverse']:
                colors.append('red')
            else:
                colors.append('gray')
        
        # Plot 1: Price levels
        bars1 = ax1.bar(range(len(scenarios)), prices, color=colors, alpha=0.7)
        ax1.axhline(y=entry_price, color='blue', linestyle='--', linewidth=2, label='Entry Price')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('CORRECTED: Test Scenarios - Price Levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add price values on bars
        for i, (bar, price) in enumerate(zip(bars1, prices)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'${price:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Plot 2: Price changes (should be realistic now)
        bars2 = ax2.bar(range(len(scenarios)), price_changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.set_ylabel('Price Change (%)')
        ax2.set_title('CORRECTED: Price Changes from Entry')
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, change) in enumerate(zip(bars2, price_changes)):
            height = bar.get_height()
            y_offset = 0.2 if height >= 0 else -0.5
            ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{change:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
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
        ax3.set_title('CORRECTED: Wallet P&L vs Thresholds')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, pnl) in enumerate(zip(bars3, wallet_pnls)):
            height = bar.get_height()
            y_offset = 0.05 if height >= 0 else -0.15
            ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{pnl:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # Plot 4: Action summary table
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for r in results:
            action = "PROFIT PROTECTION" if r['profit_protection'] else \
                    "PROFIT LOCK" if r['profit_lock'] else \
                    "REVERSE" if r['should_reverse'] else "HOLD"
            
            table_data.append([
                r['scenario'],
                f"${r['test_price']:.4f}",
                f"{r['price_change_pct']:+.1f}%",
                f"{r['wallet_pnl_pct']:+.2f}%",
                action
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Scenario', 'Price', 'Price Δ', 'Wallet P&L', 'Action'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color code the action column
        for i in range(len(table_data)):
            action = table_data[i][4]
            if action == "REVERSE":
                table[(i+1, 4)].set_facecolor('#ffcccc')  # Light red
            elif action == "PROFIT LOCK":
                table[(i+1, 4)].set_facecolor('#fff3cd')  # Light orange
            elif action == "PROFIT PROTECTION":
                table[(i+1, 4)].set_facecolor('#d4edda')  # Light green
        
        ax4.set_title('CORRECTED: Action Summary')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/corrected_reversal_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_continuous_movement(self):
        """Test continuous price movement with correct calculations"""
        print(f"\n🔄 TESTING CONTINUOUS PRICE MOVEMENT (CORRECTED)")
        print("=" * 60)
        
        entry_price = 0.085
        balance = 10000
        
        # Create realistic price range: +10% to -10%
        price_range = np.linspace(entry_price * 1.1, entry_price * 0.9, 200)
        
        results = []
        reversal_triggered = False
        reversal_point = None
        profit_lock_triggered = False
        profit_lock_point = None
        
        for i, price in enumerate(price_range):
            wallet_pnl_pct, unrealized_pnl, tokens, pos_val = self.calculate_wallet_pnl_correct(
                entry_price, price, balance)
            
            should_reverse = self.risk_manager.should_reverse_for_loss(wallet_pnl_pct)
            profit_lock = self.risk_manager.should_activate_profit_lock(wallet_pnl_pct)
            profit_protection = self.risk_manager.should_take_profit_protection(wallet_pnl_pct)
            
            # Track first triggers
            if should_reverse and not reversal_triggered:
                reversal_triggered = True
                reversal_point = i
                print(f"🔴 REVERSAL TRIGGERED at step {i}:")
                print(f"   • Price: ${price:.6f}")
                print(f"   • Price change: {((price - entry_price) / entry_price) * 100:.2f}%") 
                print(f"   • Wallet P&L: {wallet_pnl_pct:.2f}%")
            
            if profit_lock and not profit_lock_triggered:
                profit_lock_triggered = True
                profit_lock_point = i
                print(f"🔓 PROFIT LOCK TRIGGERED at step {i}:")
                print(f"   • Price: ${price:.6f}")
                print(f"   • Price change: {((price - entry_price) / entry_price) * 100:.2f}%")
                print(f"   • Wallet P&L: {wallet_pnl_pct:.2f}%")
            
            results.append({
                'step': i,
                'price': price,
                'price_change_pct': ((price - entry_price) / entry_price) * 100,
                'wallet_pnl_pct': wallet_pnl_pct,
                'unrealized_pnl': unrealized_pnl,
                'should_reverse': should_reverse,
                'profit_lock': profit_lock,
                'profit_protection': profit_protection
            })
        
        # Create plot
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price movement
        ax1.plot(df['step'], df['price'], 'b-', linewidth=2, label='ZORA Price')
        ax1.axhline(y=entry_price, color='green', linestyle='--', alpha=0.7, label='Entry Price')
        
        # Mark triggers
        if reversal_point is not None:
            ax1.scatter(reversal_point, df.iloc[reversal_point]['price'], 
                       color='red', s=200, marker='v', zorder=5, label='Reversal Triggered')
            ax1.axvline(x=reversal_point, color='red', linestyle=':', alpha=0.5)
        
        if profit_lock_point is not None:
            ax1.scatter(profit_lock_point, df.iloc[profit_lock_point]['price'], 
                       color='orange', s=200, marker='*', zorder=5, label='Profit Lock Triggered')
            ax1.axvline(x=profit_lock_point, color='orange', linestyle=':', alpha=0.5)
        
        ax1.set_title('CORRECTED: Continuous Price Movement Test')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wallet P&L
        ax2.plot(df['step'], df['wallet_pnl_pct'], 'g-', linewidth=2, label='Wallet P&L %')
        ax2.axhline(y=self.risk_manager.profit_protection_threshold, color='darkgreen', 
                   linestyle='--', linewidth=2, label='Profit Protection (2%)')
        ax2.axhline(y=self.risk_manager.profit_lock_threshold, color='orange', 
                   linestyle='--', linewidth=2, label='Profit Lock (0.5%)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=self.risk_manager.loss_reversal_threshold, color='red', 
                   linestyle='--', linewidth=2, label='Reversal Threshold (-1%)')
        
        # Shade zones
        ax2.axhspan(self.risk_manager.profit_protection_threshold, 5, alpha=0.1, color='darkgreen')
        ax2.axhspan(self.risk_manager.profit_lock_threshold, self.risk_manager.profit_protection_threshold, 
                   alpha=0.1, color='orange')
        ax2.axhspan(self.risk_manager.loss_reversal_threshold, -5, alpha=0.1, color='red')
        
        # Mark triggers
        if reversal_point is not None:
            ax2.scatter(reversal_point, df.iloc[reversal_point]['wallet_pnl_pct'], 
                       color='red', s=200, marker='v', zorder=5)
            ax2.axvline(x=reversal_point, color='red', linestyle=':', alpha=0.5)
        
        if profit_lock_point is not None:
            ax2.scatter(profit_lock_point, df.iloc[profit_lock_point]['wallet_pnl_pct'], 
                       color='orange', s=200, marker='*', zorder=5)
            ax2.axvline(x=profit_lock_point, color='orange', linestyle=':', alpha=0.5)
        
        ax2.set_title('CORRECTED: Wallet P&L Progression')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Wallet P&L (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/corrected_continuous_movement.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df, reversal_point, profit_lock_point
    
    def run_comprehensive_test(self):
        """Run all corrected tests"""
        print("🔧 CORRECTED REVERSAL THRESHOLD TEST")
        print("=" * 70)
        print("Fixed the position size calculation error!")
        print()
        
        # Test realistic scenarios
        scenario_results = self.test_realistic_reversal_scenario()
        
        # Test continuous movement  
        movement_results, reversal_point, profit_lock_point = self.test_continuous_movement()
        
        print(f"\n" + "=" * 70)
        print("✅ CORRECTED REVERSAL TESTING COMPLETE!")
        print("=" * 70)
        
        # Summary
        reversal_scenarios = [r for r in scenario_results if r['should_reverse']]
        profit_scenarios = [r for r in scenario_results if r['profit_lock'] or r['profit_protection']]
        
        print(f"📊 SUMMARY:")
        print(f"   • Scenarios tested: {len(scenario_results)}")
        print(f"   • Reversals triggered: {len(reversal_scenarios)}")
        print(f"   • Profit actions triggered: {len(profit_scenarios)}")
        print(f"   • Continuous test reversal at step: {reversal_point if reversal_point else 'None'}")
        print(f"   • Continuous test profit lock at step: {profit_lock_point if profit_lock_point else 'None'}")
        
        if reversal_scenarios:
            print(f"\n🔴 REVERSAL TRIGGERS:")
            for r in reversal_scenarios:
                print(f"   • {r['scenario']}: {r['price_change_pct']:+.2f}% price = {r['wallet_pnl_pct']:+.2f}% wallet P&L")
        
        if profit_scenarios:
            print(f"\n🟢 PROFIT TRIGGERS:")
            for r in profit_scenarios:
                action = "PROTECTION" if r['profit_protection'] else "LOCK"
                print(f"   • {r['scenario']}: {r['price_change_pct']:+.2f}% price = {r['wallet_pnl_pct']:+.2f}% wallet P&L ({action})")
        
        print(f"\n📁 Plots saved in: {self.results_dir}/")
        print("=" * 70)

if __name__ == "__main__":
    tester = CorrectedReversalTester()
    tester.run_comprehensive_test()