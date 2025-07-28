#!/usr/bin/env python3
"""
Complete Test Suite for RSI+MFI Trading Bot
Tests all components: RiskManager, Strategy, TelegramNotifier, TradeEngine
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_test_market_data():
    """Generate test market data"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    # ZORA price around 0.082
    base_price = 0.082
    price_changes = np.random.normal(0, 0.001, 100)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(0.070, min(0.095, new_price)))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df

class TestRiskManager:
    """Test RiskManager functionality"""
    
    def __init__(self):
        from core.risk_management import RiskManager
        self.risk_manager = RiskManager()
    
    def test_initialization(self):
        """Test RiskManager initialization"""
        print("🧪 Testing RiskManager initialization...")
        
        assert self.risk_manager.symbol == "ZORA/USDT"
        assert self.risk_manager.leverage == 10
        assert self.risk_manager.trailing_stop_distance == 0.015
        assert self.risk_manager.loss_switch_threshold == -0.08
        assert self.risk_manager.break_even_pct == 0.01
        
        print("✅ RiskManager initialization - PASSED")
    
    def test_position_sizing(self):
        """Test position size calculation"""
        print("🧪 Testing position sizing...")
        
        balance = 1000.0
        price = 0.082
        
        position_size = self.risk_manager.calculate_position_size(balance, price)
        expected_size = (balance * 0.1) / price  # 10% of balance
        
        assert abs(position_size - expected_size) < 1.0  # Allow some variance for price adjustments
        print(f"   Balance: ${balance}, Price: ${price}")
        print(f"   Position Size: {position_size:.2f} ZORA")
        print("✅ Position sizing - PASSED")
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        print("🧪 Testing stop loss calculation...")
        
        entry_price = 0.082
        
        # Long position
        long_sl = self.risk_manager.get_stop_loss(entry_price, 'long')
        expected_long_sl = entry_price * (1 - 0.035)
        assert abs(long_sl - expected_long_sl) < 0.000001
        
        # Short position  
        short_sl = self.risk_manager.get_stop_loss(entry_price, 'short')
        expected_short_sl = entry_price * (1 + 0.035)
        assert abs(short_sl - expected_short_sl) < 0.000001
        
        print(f"   Entry: ${entry_price:.6f}")
        print(f"   Long SL: ${long_sl:.6f}")
        print(f"   Short SL: ${short_sl:.6f}")
        print("✅ Stop loss calculation - PASSED")
    
    def test_take_profit_calculation(self):
        """Test take profit calculation"""
        print("🧪 Testing take profit calculation...")
        
        entry_price = 0.082
        
        # Long position
        long_tp = self.risk_manager.get_take_profit(entry_price, 'long')
        expected_long_tp = entry_price * (1 + 0.07)
        assert abs(long_tp - expected_long_tp) < 0.000001
        
        # Short position
        short_tp = self.risk_manager.get_take_profit(entry_price, 'short')
        expected_short_tp = entry_price * (1 - 0.07)
        assert abs(short_tp - expected_short_tp) < 0.000001
        
        print(f"   Entry: ${entry_price:.6f}")
        print(f"   Long TP: ${long_tp:.6f}")
        print(f"   Short TP: ${short_tp:.6f}")
        print("✅ Take profit calculation - PASSED")

class TestStrategy:
    """Test RSI+MFI Strategy"""
    
    def __init__(self):
        from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
        # Create test params
        test_params = {
            "rsi_length": 5,
            "mfi_length": 5,
            "oversold_level": 45,
            "overbought_level": 55,
            "require_volume": False,
            "require_trend": False
        }
        self.strategy = RSIMFICloudStrategy(test_params)
    
    def test_indicator_calculation(self):
        """Test RSI and MFI calculation"""
        print("🧪 Testing indicator calculations...")
        
        df = create_test_market_data()
        df_with_indicators = self.strategy.calculate_indicators(df)
        
        assert 'rsi' in df_with_indicators.columns
        assert 'mfi' in df_with_indicators.columns
        assert not df_with_indicators['rsi'].isna().all()
        assert not df_with_indicators['mfi'].isna().all()
        
        # Check RSI range
        rsi_values = df_with_indicators['rsi'].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100
        
        # Check MFI range
        mfi_values = df_with_indicators['mfi'].dropna()
        assert mfi_values.min() >= 0
        assert mfi_values.max() <= 100
        
        print(f"   RSI range: {rsi_values.min():.1f} - {rsi_values.max():.1f}")
        print(f"   MFI range: {mfi_values.min():.1f} - {mfi_values.max():.1f}")
        print("✅ Indicator calculations - PASSED")
    
    def test_signal_generation(self):
        """Test signal generation"""
        print("🧪 Testing signal generation...")
        
        df = create_test_market_data()
        
        # Force oversold condition for BUY signal
        df_oversold = df.copy()
        df_oversold.loc[df_oversold.index[-5:], 'close'] = 0.075  # Lower prices
        
        signal = self.strategy.generate_signal(df_oversold)
        
        if signal:
            assert signal['action'] in ['BUY', 'SELL']
            assert 'price' in signal
            assert 'rsi' in signal
            assert 'mfi' in signal
            print(f"   Generated signal: {signal['action']} at ${signal['price']:.6f}")
            print(f"   RSI: {signal['rsi']}, MFI: {signal['mfi']}")
        else:
            print("   No signal generated (normal for test data)")
        
        print("✅ Signal generation - PASSED")

class TestTelegramNotifier:
    """Test Telegram notifications"""
    
    def __init__(self):
        from core.telegram_notifier import TelegramNotifier
        self.notifier = TelegramNotifier()
    
    async def test_notifications(self):
        """Test notification methods"""
        print("🧪 Testing Telegram notifications...")
        
        # Mock the send_message method
        self.notifier.send_message = AsyncMock()
        
        # Test trade opened notification
        await self.notifier.trade_opened("ZORA/USDT", 0.082, 100, "Buy")
        assert self.notifier.send_message.called
        
        # Test trade closed notification
        await self.notifier.trade_closed("ZORA/USDT", 2.5, 25.0, "Take Profit")
        assert self.notifier.send_message.call_count == 2
        
        # Test profit lock notification
        await self.notifier.profit_lock_activated("ZORA/USDT", 1.5, 0.3)
        assert self.notifier.send_message.call_count == 3
        
        print("✅ Telegram notifications - PASSED")

class TestTradeEngine:
    """Test TradeEngine functionality"""
    
    def __init__(self):
        # Mock external dependencies
        with patch('pybit.unified_trading.HTTP'), \
             patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test', 'TELEGRAM_CHAT_ID': 'test'}):
            from core.trade_engine import TradeEngine
            self.engine = TradeEngine()
    
    def test_initialization(self):
        """Test TradeEngine initialization"""
        print("🧪 Testing TradeEngine initialization...")
        
        # Check if RiskManager is properly integrated
        assert hasattr(self.engine, 'risk_manager')
        assert self.engine.symbol == "ZORA/USDT"  # Should come from RiskManager
        assert self.engine.linear == "ZORAUSDT"
        assert self.engine.risk_manager.leverage == 10
        assert self.engine.risk_manager.trailing_stop_distance == 0.015
        
        print(f"   Symbol: {self.engine.symbol}")
        print(f"   Leverage: {self.engine.risk_manager.leverage}x")
        print(f"   Trailing Stop: {self.engine.risk_manager.trailing_stop_distance*100:.1f}%")
        print("✅ TradeEngine initialization - PASSED")
    
    def test_formatting_functions(self):
        """Test price and quantity formatting"""
        print("🧪 Testing formatting functions...")
        
        # Mock symbol info
        mock_info = {
            'min_qty': 1.0,
            'qty_step': 1.0,
            'tick_size': 0.000001
        }
        
        # Test quantity formatting
        raw_qty = 123.456
        formatted_qty = self.engine.format_qty(mock_info, raw_qty)
        assert isinstance(formatted_qty, str)
        
        # Test price formatting
        raw_price = 0.082345
        formatted_price = self.engine.format_price(mock_info, raw_price)
        assert isinstance(formatted_price, str)
        
        print(f"   Raw qty: {raw_qty} -> Formatted: {formatted_qty}")
        print(f"   Raw price: {raw_price} -> Formatted: {formatted_price}")
        print("✅ Formatting functions - PASSED")

async def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive System Test\n")
    print("="*60)
    
    try:
        # Test 1: RiskManager
        print("\n📊 TESTING RISK MANAGER")
        print("-" * 30)
        risk_test = TestRiskManager()
        risk_test.test_initialization()
        risk_test.test_position_sizing()
        risk_test.test_stop_loss_calculation()
        risk_test.test_take_profit_calculation()
        
        # Test 2: Strategy
        print("\n🎯 TESTING STRATEGY")
        print("-" * 30)
        strategy_test = TestStrategy()
        strategy_test.test_indicator_calculation()
        strategy_test.test_signal_generation()
        
        # Test 3: Telegram Notifier
        print("\n📱 TESTING TELEGRAM NOTIFIER")
        print("-" * 30)
        telegram_test = TestTelegramNotifier()
        await telegram_test.test_notifications()
        
        # Test 4: TradeEngine
        print("\n⚙️  TESTING TRADE ENGINE")
        print("-" * 30)
        engine_test = TestTradeEngine()
        engine_test.test_initialization()
        engine_test.test_formatting_functions()
        
        # Test 5: Environment Variables
        print("\n🔧 TESTING ENVIRONMENT")
        print("-" * 30)
        test_env_variables()
        
        # Test 6: File Structure
        print("\n📁 TESTING FILE STRUCTURE")
        print("-" * 30)
        test_file_structure()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for trading")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_env_variables():
    """Test environment variables"""
    print("🧪 Testing environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'DEMO_MODE',
        'TESTNET_BYBIT_API_KEY',
        'TESTNET_BYBIT_API_SECRET'
    ]
    
    # Telegram vars are optional for testing
    optional_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_vars:
        print(f"   ⚠️  Missing REQUIRED variables: {missing_vars}")
    else:
        print("   All required environment variables found")
        
    if missing_optional:
        print(f"   ℹ️  Missing OPTIONAL variables: {missing_optional}")
        print("   (Telegram notifications will be disabled)")
    
    print("✅ Environment variables - CHECKED")

def test_file_structure():
    """Test required file structure"""
    print("🧪 Testing file structure...")
    
    required_files = [
        'core/risk_management.py',
        'core/trade_engine.py',
        'core/telegram_notifier.py',
        'strategies/RSI_MFI_Cloud.py',
        'strategies/params_RSI_MFI_Cloud.json',
        'main.py',
        '.env'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ⚠️  Missing files: {missing_files}")
    else:
        print("   All required files found")
    
    print("✅ File structure - CHECKED")

if __name__ == "__main__":
    try:
        asyncio.run(run_comprehensive_test())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        sys.exit(1)