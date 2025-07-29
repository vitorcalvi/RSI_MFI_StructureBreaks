#!/usr/bin/env python3

"""
Simple script to run the HFT optimizer
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the fixed optimizer
from backtrader import grid_search_optimization

def main():
    print("="*60)
    print("HFT OPTIMIZER FOR ZORA/USDT 5M")
    print("="*60)
    print()
    
    try:
        # Run the optimization
        best_params, results = grid_search_optimization()
        
        if best_params:
            print("\n" + "="*40)
            print("✅ OPTIMIZATION SUCCESSFUL!")
            print("="*40)
            print("\n📁 Files created:")
            print("  • optimized_params.json")
            print("  • ensemble_params.json")
            
            print("\n🚀 Next steps:")
            print("  1. Update your strategy with optimized_params.json")
            print("  2. Test on live data")
            print("  3. Consider ensemble trading with top 5 params")
            
        else:
            print("\n❌ No valid parameters found")
            print("Try adjusting the parameter bounds in fixed_optimizer.py")
            
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print("Make sure RSI_MFI_Cloud.py is in the strategies folder")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)