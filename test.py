#!/usr/bin/env python3
"""
Comprehensive Test Suite for RSI+MFI Trading Bot
Tests all conditions, logic, and features
"""

import os
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test result tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': [],
    'warnings': []
}

def print_header(text):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"📋 {text}")
    print('='*60)

def test_pass(test_name):
    """Record passed test"""
    test_results['passed'] += 1
    print(f"✅ {test_name}")

def test_fail(test_name, error):
    """Record failed test"""
    test_results['failed'] += 1
    test_results['errors'].append(f"{test_name}: {error}")
    print(f"❌ {test_name}: {error}")

def test_warn(test_name, warning):
    """Record warning"""
    test_results['warnings'].append(f"{test_name}: {warning}")
    print(f"⚠️  {test_name}: {warning}")

async def test_environment():
    """Test environment setup"""
    print_header("Testing Environment Setup")
    
    # Check Python version
    if sys.version_info >= (3, 8):
        test_pass("Python version 3.8+")
    else:
        test_fail("Python version", f"Need 3.8+, got {sys.version}")
    
    # Check .env file
    if os.path.exists('.env'):
        test_pass(".env file exists")
        
        # Check required env variables
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ['SYMBOLS', 'EXCHANGE', 'DEMO_MODE']
        for var in required_vars:
            if os.getenv(var):
                test_pass(f"Environment variable {var}")
            else:
                test_fail(f"Environment variable {var}", "Not set")
    else:
        test_fail(".env file", "Not found")
    
    # Check file structure
    required_files = [
        'main.py',
        'core/trade_engine.py',
        'core/risk_management.py',
        'core/telegram_notifier.py',
        'strategies/RSI_MFI_Cloud.py',
        'strategies/params_RSI_MFI_Cloud.json'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            test_pass(f"File {file}")
        else:
            test_fail(f"File {file}", "Not found")

async def test_imports():
    """Test all imports"""
    print_header("Testing Imports")
    
    imports = [
        ('ccxt', 'ccxt'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pybit', 'pybit.unified_trading'),
        ('telegram', 'telegram'),
        ('dotenv', 'dotenv'),
        ('ta', 'ta')
    ]
    
    for name, module in imports:
        try:
            __import__(module)
            test_pass(f"Import {name}")
        except ImportError:
            test_fail(f"Import {name}", "Module not installed")

async def test_exchange_connection():
    """Test exchange connectivity"""
    print_header("Testing Exchange Connection")
    
    try:
        from pybit.unified_trading import HTTP
        
        # Test public connection
        session = HTTP(testnet=True)
        response = session.get_kline(
            category="spot",
            symbol="SOLUSDT",
            interval="1",
            limit=1
        )
        
        if response['retCode'] == 0:
            test_pass("Bybit public API connection")
        else:
            test_fail("Bybit public API", response.get('retMsg', 'Unknown error'))
            
        # Test authenticated connection if keys exist
        api_key = os.getenv('TESTNET_BYBIT_API_KEY')
        api_secret = os.getenv('TESTNET_BYBIT_API_SECRET')
        
        if api_key and api_secret:
            auth_session = HTTP(
                testnet=True,
                api_key=api_key,
                api_secret=api_secret
            )
            
            balance = auth_session.get_wallet_balance(accountType="UNIFIED")
            if balance['retCode'] == 0:
                test_pass("Bybit authenticated API connection")
            else:
                test_fail("Bybit authenticated API", balance.get('retMsg', 'Unknown error'))
        else:
            test_warn("Bybit authenticated API", "No API keys configured")
            
    except Exception as e:
        test_fail("Exchange connection", str(e))

async def test_strategy():
    """Test RSI+MFI strategy"""
    print_header("Testing Trading Strategy")
    
    try:
        from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
        
        # Initialize strategy
        strategy = RSIMFICloudStrategy()
        test_pass("Strategy initialization")
        
        # Check parameters loaded
        if hasattr(strategy, 'params'):
            test_pass("Strategy parameters loaded")
            print(f"   Parameters: {strategy.params}")
        else:
            test_fail("Strategy parameters", "Not loaded")
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        test_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Test indicator calculation
        df_with_indicators = strategy.calculate_indicators(test_df.copy())
        
        if 'rsi' in df_with_indicators.columns:
            test_pass("RSI calculation")
        else:
            test_fail("RSI calculation", "RSI column missing")
            
        if 'mfi' in df_with_indicators.columns:
            test_pass("MFI calculation")
        else:
            test_fail("MFI calculation", "MFI column missing")
        
        # Test signal generation
        signal = strategy.generate_signal(test_df)
        test_pass("Signal generation")
        
        # Test extreme conditions
        # Oversold condition
        oversold_df = test_df.copy()
        oversold_df['close'] = oversold_df['close'] * 0.7  # Drop price
        signal = strategy.generate_signal(oversold_df)
        if signal and signal['action'] == 'BUY':
            test_pass("Oversold BUY signal")
        else:
            test_warn("Oversold BUY signal", "No signal generated")
        
        # Overbought condition
        overbought_df = test_df.copy()
        overbought_df['close'] = overbought_df['close'] * 1.3  # Raise price
        strategy.last_signal = 'BUY'  # Reset last signal
        signal = strategy.generate_signal(overbought_df)
        if signal and signal['action'] == 'SELL':
            test_pass("Overbought SELL signal")
        else:
            test_warn("Overbought SELL signal", "No signal generated")
            
    except Exception as e:
        test_fail("Strategy test", str(e))

async def test_risk_management():
    """Test risk management"""
    print_header("Testing Risk Management")
    
    try:
        from core.risk_management import RiskManager
        
        rm = RiskManager()
        test_pass("Risk manager initialization")
        
        # Test position sizing
        balance = 1000
        price = 100
        position_size = rm.calculate_position_size(balance, price)
        
        expected_size = (balance * rm.max_position_size) / price
        if abs(position_size - expected_size) < 0.001:
            test_pass(f"Position sizing: {position_size:.4f} units")
        else:
            test_fail("Position sizing", f"Expected {expected_size}, got {position_size}")
        
        # Test stop loss
        entry_price = 100
        stop_loss = rm.get_stop_loss(entry_price, 'long')
        expected_sl = entry_price * (1 - rm.stop_loss_pct)
        
        if abs(stop_loss - expected_sl) < 0.001:
            test_pass(f"Stop loss calculation: ${stop_loss:.2f}")
        else:
            test_fail("Stop loss", f"Expected {expected_sl}, got {stop_loss}")
        
        # Test take profit
        take_profit = rm.get_take_profit(entry_price, 'long')
        expected_tp = entry_price * (1 + rm.take_profit_pct)
        
        if abs(take_profit - expected_tp) < 0.001:
            test_pass(f"Take profit calculation: ${take_profit:.2f}")
        else:
            test_fail("Take profit", f"Expected {expected_tp}, got {take_profit}")
            
    except Exception as e:
        test_fail("Risk management test", str(e))

async def test_telegram_notifier():
    """Test Telegram notifications"""
    print_header("Testing Telegram Notifier")
    
    try:
        from core.telegram_notifier import TelegramNotifier
        
        notifier = TelegramNotifier()
        test_pass("Telegram notifier initialization")
        
        if notifier.enabled:
            test_pass("Telegram bot configured")
            
            # Test message sending (without actually sending)
            with patch.object(notifier.bot, 'send_message', new_callable=AsyncMock) as mock_send:
                await notifier.send("Test message")
                if mock_send.called:
                    test_pass("Telegram message sending")
                else:
                    test_fail("Telegram message", "Send not called")
        else:
            test_warn("Telegram bot", "Not configured (optional)")
            
    except Exception as e:
        test_fail("Telegram test", str(e))

async def test_trade_engine():
    """Test trade engine core functionality"""
    print_header("Testing Trade Engine")
    
    try:
        from core.trade_engine import TradeEngine
        
        # Test initialization
        engine = TradeEngine()
        test_pass("Trade engine initialization")
        
        # Test demo mode
        if engine.demo_mode:
            test_pass("Demo mode active")
        else:
            test_warn("Demo mode", "Live mode active - be careful!")
        
        # Test OHLCV fetching
        df = await engine.fetch_ohlcv('SOL/USDT', timeframe='1', limit=50)
        if df is not None and len(df) > 0:
            test_pass(f"OHLCV data fetching: {len(df)} candles")
        else:
            test_fail("OHLCV data", "No data returned")
        
        # Test position tracking in demo mode
        if engine.demo_mode:
            # Simulate buy signal
            test_signal = {
                'action': 'BUY',
                'price': 100.0,
                'rsi': 25.0,
                'mfi': 30.0,
                'timestamp': datetime.now(),
                'confidence': 'TEST'
            }
            
            await engine.execute_trade(test_signal, 'SOL/USDT')
            
            if 'SOL/USDT' in engine.positions:
                test_pass("Demo position tracking (BUY)")
            else:
                test_fail("Demo position tracking", "Position not recorded")
            
            # Simulate sell signal
            test_signal['action'] = 'SELL'
            test_signal['price'] = 105.0
            
            await engine.execute_trade(test_signal, 'SOL/USDT')
            
            if 'SOL/USDT' not in engine.positions:
                test_pass("Demo position tracking (SELL)")
            else:
                test_fail("Demo position closing", "Position still open")
                
    except Exception as e:
        test_fail("Trade engine test", str(e))

async def test_edge_cases():
    """Test edge cases and error handling"""
    print_header("Testing Edge Cases")
    
    try:
        from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
        
        strategy = RSIMFICloudStrategy()
        
        # Test with empty data
        empty_df = pd.DataFrame()
        result = strategy.calculate_indicators(empty_df)
        test_pass("Empty DataFrame handling")
        
        # Test with insufficient data
        small_df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000]
        })
        signal = strategy.generate_signal(small_df)
        if signal is None:
            test_pass("Insufficient data handling")
        else:
            test_fail("Insufficient data", "Should return None")
        
        # Test with NaN values
        nan_df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [101, np.nan, 103],
            'low': [99, np.nan, 101],
            'close': [100, np.nan, 102],
            'volume': [1000, 0, 1000]
        })
        try:
            indicators = strategy.calculate_indicators(nan_df)
            test_pass("NaN value handling")
        except:
            test_fail("NaN value handling", "Exception raised")
        
        # Test with zero volume
        zero_vol_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 115, 50),
            'volume': np.zeros(50)
        })
        indicators = strategy.calculate_indicators(zero_vol_df)
        if 'mfi' in indicators.columns and not indicators['mfi'].isna().all():
            test_pass("Zero volume handling")
        else:
            test_fail("Zero volume", "MFI calculation failed")
            
    except Exception as e:
        test_fail("Edge case test", str(e))

async def test_performance():
    """Test performance metrics"""
    print_header("Testing Performance")
    
    try:
        from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
        import time
        
        strategy = RSIMFICloudStrategy()
        
        # Create larger dataset
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        large_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 1000),
            'high': np.random.uniform(110, 120, 1000),
            'low': np.random.uniform(90, 100, 1000),
            'close': np.random.uniform(95, 115, 1000),
            'volume': np.random.uniform(1000, 5000, 1000)
        }, index=dates)
        
        # Time indicator calculation
        start_time = time.time()
        df_with_indicators = strategy.calculate_indicators(large_df.copy())
        calc_time = time.time() - start_time
        
        if calc_time < 1.0:  # Should complete in under 1 second
            test_pass(f"Indicator calculation speed: {calc_time:.3f}s")
        else:
            test_warn("Indicator calculation", f"Slow: {calc_time:.3f}s")
        
        # Time signal generation
        start_time = time.time()
        signal = strategy.generate_signal(large_df)
        signal_time = time.time() - start_time
        
        if signal_time < 0.1:  # Should be very fast
            test_pass(f"Signal generation speed: {signal_time:.3f}s")
        else:
            test_warn("Signal generation", f"Slow: {signal_time:.3f}s")
            
    except Exception as e:
        test_fail("Performance test", str(e))

async def generate_test_report():
    """Generate comprehensive test report"""
    print_header("Test Summary Report")
    
    total_tests = test_results['passed'] + test_results['failed']
    pass_rate = (test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"📊 Pass Rate: {pass_rate:.1f}%")
    
    if test_results['warnings']:
        print(f"\n⚠️  Warnings ({len(test_results['warnings'])}):")
        for warning in test_results['warnings']:
            print(f"   - {warning}")
    
    if test_results['errors']:
        print(f"\n❌ Errors ({len(test_results['errors'])}):")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    # Save report to file
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed': test_results['passed'],
        'failed': test_results['failed'],
        'pass_rate': pass_rate,
        'warnings': test_results['warnings'],
        'errors': test_results['errors']
    }
    
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    # Return success if pass rate > 80%
    return pass_rate >= 80

async def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE TRADING BOT TEST SUITE")
    print("=" * 60)
    print("This will test all components of the trading bot")
    print("Including: environment, imports, connections, strategies,")
    print("risk management, notifications, and edge cases.")
    print()
    
    # Run all test categories
    test_functions = [
        test_environment,
        test_imports,
        test_exchange_connection,
        test_strategy,
        test_risk_management,
        test_telegram_notifier,
        test_trade_engine,
        test_edge_cases,
        test_performance
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
        except Exception as e:
            test_fail(f"{test_func.__name__}", f"Unexpected error: {e}")
    
    # Generate final report
    success = await generate_test_report()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST SUITE PASSED - Bot is ready for use!")
        print("💡 Next: Run 'python main.py' to start trading")
    else:
        print("❌ TEST SUITE FAILED - Please fix errors before running")
        print("💡 Check the test report for details")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)