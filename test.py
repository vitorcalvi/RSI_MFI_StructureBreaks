#!/usr/bin/env python3
"""
ZORA Trading Bot - Master Test Runner
Run all TradeEngine feature tests individually or together
"""

import os
import sys
import asyncio
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_header():
    """Print master test header"""
    print("🧪 ZORA Trading Bot - Master Test Suite")
    print("=" * 70)
    print("🚀 Comprehensive TradeEngine Feature Testing")
    print("=" * 70)

def print_menu():
    """Print test selection menu"""
    print("\n📋 Available Tests:")
    print("=" * 50)
    print("1. 🔒 Trailing Stop Profit Locker Test")
    print("2. 🔀 Position Reversal Test")
    print("3. 💰 Profit Protection Test")
    print("4. ⏳ Signal Cooldown Test")
    print("5. 🧪 Complete System Test (all components)")
    print("6. 🚀 Run ALL Tests Sequentially")
    print("0. ❌ Exit")
    print("=" * 50)

async def run_trailing_stop_test():
    """Run trailing stop profit locker test"""
    try:
        print("\n🔒 Running Trailing Stop Profit Locker Test...")
        print("-" * 60)
        
        # Use the comprehensive test file instead
        import comprehensive_features_test
        tester = comprehensive_features_test.ComprehensiveTester()
        results = tester.test_trailing_stop_profit_locker()
        tester.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Trailing Stop Test Error: {e}")
        return False

async def run_position_reversal_test():
    """Run position reversal test"""
    try:
        print("\n🔀 Running Position Reversal Test...")
        print("-" * 60)
        
        import comprehensive_features_test
        tester = comprehensive_features_test.ComprehensiveTester()
        results = tester.test_position_reversal()
        tester.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Position Reversal Test Error: {e}")
        return False

async def run_profit_protection_test():
    """Run profit protection test"""
    try:
        print("\n💰 Running Profit Protection Test...")
        print("-" * 60)
        
        import comprehensive_features_test
        tester = comprehensive_features_test.ComprehensiveTester()
        results = tester.test_profit_protection()
        tester.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Profit Protection Test Error: {e}")
        return False

async def run_signal_cooldown_test():
    """Run signal cooldown test"""
    try:
        print("\n⏳ Running Signal Cooldown Test...")
        print("-" * 60)
        
        import comprehensive_features_test
        tester = comprehensive_features_test.ComprehensiveTester()
        results = tester.test_signal_cooldown()
        tester.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Signal Cooldown Test Error: {e}")
        return False

async def run_complete_system_test():
    """Run the original comprehensive system test"""
    try:
        print("\n🧪 Running Complete System Test...")
        print("-" * 60)
        
        # Run the original comprehensive test from test.py
        import test
        test_suite = test.TestSuite()
        await test_suite.run_all_tests()
        
        return True
        
    except Exception as e:
        print(f"❌ Complete System Test Error: {e}")
        return False

async def run_all_tests():
    """Run all tests sequentially"""
    print("\n🚀 Running ALL Tests Sequentially...")
    print("=" * 70)
    
    tests = [
        ("Trailing Stop Profit Locker", run_trailing_stop_test),
        ("Position Reversal", run_position_reversal_test),
        ("Profit Protection", run_profit_protection_test),
        ("Signal Cooldown", run_signal_cooldown_test),
        ("Complete System", run_complete_system_test)
    ]
    
    results = {}
    start_time = datetime.now()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAILED: {test_name} - {e}")
        
        print("-" * 70)
    
    # Print final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} - {test_name}")
    
    print(f"\n📈 Overall Results:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    print(f"   📈 Success Rate: {(passed/total)*100:.1f}%")
    print(f"   ⏱️ Total Duration: {duration:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 ZORA Trading Bot is ready for deployment!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Please review and fix issues before deployment.")
    
    print("=" * 70)

def print_test_descriptions():
    """Print detailed descriptions of each test"""
    print("\n📖 Test Descriptions:")
    print("=" * 50)
    
    descriptions = {
        "🔒 Trailing Stop Profit Locker": [
            "Tests the trailing stop mechanism that protects profits",
            "Verifies stop moves up with favorable price movement",
            "Ensures profits are locked when profit threshold reached",
            "Validates that losses don't occur after profit lock activation"
        ],
        "🔀 Position Reversal": [
            "Tests position reversal logic for opposite signals",
            "Verifies profit protection reversal at 4% threshold",
            "Tests losing position reversal at -5% threshold",
            "Validates signal ignoring for moderate P&L levels"
        ],
        "💰 Profit Protection": [
            "Tests profit taking at 4% account profit threshold",
            "Verifies cooldown activation after profit protection",
            "Tests prevention of new positions during cooldown",
            "Validates risk management discipline"
        ],
        "⏳ Signal Cooldown": [
            "Tests signal cooldown mechanism (5 bars)",
            "Verifies prevention of overtrading",
            "Tests signal quality improvement through filtering",
            "Validates timing control between signals"
        ],
        "🧪 Complete System": [
            "Tests all components together",
            "Verifies component integration",
            "Tests edge cases and error handling",
            "Validates overall system reliability"
        ]
    }
    
    for test_name, features in descriptions.items():
        print(f"\n{test_name}:")
        for feature in features:
            print(f"   • {feature}")
    
    print("=" * 50)

async def main():
    """Main test runner interface"""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n🎯 Select test to run (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 Exiting Master Test Runner")
                break
            elif choice == '1':
                await run_trailing_stop_test()
            elif choice == '2':
                await run_position_reversal_test()
            elif choice == '3':
                await run_profit_protection_test()
            elif choice == '4':
                await run_signal_cooldown_test()
            elif choice == '5':
                await run_complete_system_test()
            elif choice == '6':
                await run_all_tests()
            elif choice.lower() == 'help':
                print_test_descriptions()
            else:
                print("❌ Invalid choice. Please select 0-6 or type 'help' for descriptions.")
            
            if choice in ['1', '2', '3', '4', '5', '6']:
                input("\n⏸️ Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting Master Test Runner")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)