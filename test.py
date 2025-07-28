#!/usr/bin/env python3
"""
Comprehensive Test Suite for ZORA Trading Bot
Tests all components: Strategy, Risk Management, Data Processing, etc.
"""

import os
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create mock params file for testing
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

def generate_test_data(length=100, start_price=0.082, volatility=0.05):
    """Generate realistic OHLCV test data for ZORA"""
    dates = pd.date_range(start=datetime.now() - timedelta(hours=length), periods=length, freq='5min')
    
    # Generate realistic price movements
    returns = np.random.normal(0, volatility, length)
    prices = [start_price]
    
    for r in returns[1:]:
        new_price = prices[-1] * (1 + r)
        new_price = max(0.001, new_price)  # Prevent negative prices
        prices.append(new_price)
    
    # Generate OHLC from prices
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
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

class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.params_file = None
        
    def setup(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        self.params_file = create_test_params()
        print(f"✅ Created test params: {self.params_file}")
        
    def cleanup(self):
        """Cleanup test files"""
        if self.params_file and os.path.exists(self.params_file):
            os.remove(self.params_file)
            print(f"🧹 Cleaned up: {self.params_file}")
    
    def assert_test(self, condition, test_name, details=""):
        """Assert test result"""
        if condition:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name} - {details}")
            self.failed += 1
    
    def test_risk_manager(self):
        """Test RiskManager functionality"""
        print("\n📊 Testing RiskManager...")
        
        try:
            from core.risk_management import RiskManager
            rm = RiskManager()
            
            # Test basic parameters
            self.assert_test(rm.symbol == "ZORA/USDT", "Symbol configuration")
            self.assert_test(rm.leverage == 10, "Leverage setting")
            self.assert_test(rm.trailing_stop_distance == 0.008, "Trailing stop distance")
            
            # Test position size calculation
            balance = 1000
            price = 0.082
            pos_size = rm.calculate_position_size(balance, price)
            expected_size = (balance * 0.1) / price  # No multiplier for price > 0.05
            
            self.assert_test(abs(pos_size - expected_size) < 0.1, 
                           "Position size calculation", 
                           f"Got {pos_size}, expected ~{expected_size}")
            
            # Test stop loss calculation
            entry_price = 0.082
            long_sl = rm.get_stop_loss(entry_price, 'long')
            short_sl = rm.get_stop_loss(entry_price, 'short')
            
            expected_long_sl = entry_price * (1 - rm.stop_loss_pct)
            expected_short_sl = entry_price * (1 + rm.stop_loss_pct)
            
            self.assert_test(abs(long_sl - expected_long_sl) < 0.0001, "Long stop loss")
            self.assert_test(abs(short_sl - expected_short_sl) < 0.0001, "Short stop loss")
            
            # Test take profit calculation
            long_tp = rm.get_take_profit(entry_price, 'long')
            short_tp = rm.get_take_profit(entry_price, 'short')
            
            expected_long_tp = entry_price * (1 + rm.take_profit_pct)
            expected_short_tp = entry_price * (1 - rm.take_profit_pct)
            
            self.assert_test(abs(long_tp - expected_long_tp) < 0.0001, "Long take profit")
            self.assert_test(abs(short_tp - expected_short_tp) < 0.0001, "Short take profit")
            
        except Exception as e:
            self.assert_test(False, "RiskManager import/creation", str(e))
    
    def test_strategy_indicators(self):
        """Test strategy indicator calculations"""
        print("\n🎯 Testing Strategy Indicators...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            # Generate test data
            df = generate_test_data(50)
            
            # Test RSI calculation
            rsi = strategy.calculate_rsi(df['close'])
            self.assert_test(len(rsi) == len(df), "RSI length matches data")
            self.assert_test(rsi.min() >= 0 and rsi.max() <= 100, "RSI bounds check")
            self.assert_test(not rsi.isna().all(), "RSI not all NaN")
            
            # Test MFI calculation
            mfi = strategy.calculate_mfi(df['high'], df['low'], df['close'])
            self.assert_test(len(mfi) == len(df), "MFI length matches data")
            self.assert_test(mfi.min() >= 0 and mfi.max() <= 100, "MFI bounds check")
            self.assert_test(not mfi.isna().all(), "MFI not all NaN")
            
            # Test trend calculation
            trend = strategy.calculate_trend(df['close'])
            self.assert_test(len(trend) == len(df), "Trend length matches data")
            valid_trends = ['UP', 'DOWN', 'SIDEWAYS']
            self.assert_test(all(t in valid_trends for t in trend), "Valid trend values")
            
            # Test ATR calculation
            atr = strategy.calculate_atr(df)
            self.assert_test(len(atr) == len(df), "ATR length matches data")
            self.assert_test((atr >= 0).all(), "ATR non-negative")
            
            # Test full indicator calculation
            df_with_indicators = strategy.calculate_indicators(df)
            required_cols = ['rsi', 'mfi', 'trend', 'price_change', 'atr']
            
            for col in required_cols:
                self.assert_test(col in df_with_indicators.columns, f"Has {col} column")
            
        except Exception as e:
            self.assert_test(False, "Strategy indicator calculation", str(e))
    
    def test_signal_generation(self):
        """Test signal generation logic"""
        print("\n📡 Testing Signal Generation...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            # Test with insufficient data
            small_df = generate_test_data(10)
            signal = strategy.generate_signal(small_df)
            self.assert_test(signal is None, "No signal with insufficient data")
            
            # Test with sufficient data
            df = generate_test_data(100)
            
            # Force oversold conditions for BUY signal
            df_oversold = df.copy()
            df_oversold.loc[df_oversold.index[-10:], 'close'] *= 0.9  # Drop price to create oversold
            
            # Test signal generation
            signal = strategy.generate_signal(df_oversold)
            
            # Signal should be dict or None
            if signal is not None:
                required_keys = ['action', 'price', 'rsi', 'mfi', 'trend', 'timestamp']
                for key in required_keys:
                    self.assert_test(key in signal, f"Signal has {key}")
                
                self.assert_test(signal['action'] in ['BUY', 'SELL', 'CLOSE'], "Valid signal action")
                self.assert_test(isinstance(signal['price'], (int, float)), "Price is numeric")
            
            # Test signal cooldown
            first_signal = strategy.generate_signal(df)
            if first_signal:
                second_signal = strategy.generate_signal(df)  # Should be None due to cooldown
                self.assert_test(second_signal is None, "Signal cooldown working")
            
        except Exception as e:
            self.assert_test(False, "Signal generation", str(e))
    
    def test_atr_risk_management(self):
        """Test ATR-based risk management"""
        print("\n🔒 Testing ATR Risk Management...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            df = generate_test_data(50)
            df_with_indicators = strategy.calculate_indicators(df)
            
            current_price = df_with_indicators['close'].iloc[-1]
            current_atr = df_with_indicators['atr'].iloc[-1]
            
            # Test position setting
            strategy.set_position('LONG', current_price, current_atr)
            self.assert_test(strategy.position_type == 'LONG', "Position type set")
            self.assert_test(strategy.entry_price == current_price, "Entry price set")
            self.assert_test(strategy.trailing_stop is not None, "Trailing stop set")
            
            expected_stop = current_price - (current_atr * strategy.atr_multiplier)
            self.assert_test(abs(strategy.trailing_stop - expected_stop) < 0.0001, 
                           "Correct trailing stop calculation")
            
            # Test trailing stop update
            new_price = current_price * 1.02  # 2% higher
            strategy.update_trailing_stop(new_price, current_atr)
            
            # Stop should move up for profitable long position
            new_expected_stop = new_price - (current_atr * strategy.atr_multiplier)
            self.assert_test(strategy.trailing_stop >= expected_stop, "Trailing stop moved up")
            
            # Test stop hit detection
            hit_price = strategy.trailing_stop - 0.001  # Below stop
            self.assert_test(strategy.check_stop_hit(hit_price), "Stop hit detection")
            
            safe_price = strategy.trailing_stop + 0.001  # Above stop
            self.assert_test(not strategy.check_stop_hit(safe_price), "Stop not hit detection")
            
            # Test position reset
            strategy.reset_position()
            self.assert_test(strategy.position_type is None, "Position type reset")
            self.assert_test(strategy.trailing_stop is None, "Trailing stop reset")
            
        except Exception as e:
            self.assert_test(False, "ATR risk management", str(e))
   
    async def test_telegram_notifier(self):
        """Test Telegram notifier (mock)"""
        print("\n📱 Testing Telegram Notifier...")
        
        try:
            # Mock environment variables
            with patch.dict(os.environ, {
                'TELEGRAM_BOT_TOKEN': 'mock_token',
                'TELEGRAM_CHAT_ID': 'mock_chat_id'
            }):
                with patch('telegram.Bot') as mock_bot:
                    from core.telegram_notifier import TelegramNotifier
                    
                    notifier = TelegramNotifier()
                    self.assert_test(notifier.enabled, "Notifier enabled with mock credentials")
                    
                    # Test message sending (mock)
                    mock_bot_instance = AsyncMock()
                    notifier.bot = mock_bot_instance
                    
                    await notifier.trade_opened("ZORA/USDT", 0.082, 1000, "Buy")
                    self.assert_test(mock_bot_instance.send_message.called, "Trade opened notification sent")
                    
                    await notifier.trade_closed("ZORA/USDT", 5.2, 10.5, "Take Profit")
                    self.assert_test(mock_bot_instance.send_message.call_count >= 2, "Trade closed notification sent")
                    
        except Exception as e:
            self.assert_test(False, "Telegram notifier", str(e))
    
    def test_data_processing(self):
        """Test data processing and edge cases"""
        print("\n📈 Testing Data Processing...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            # Test with NaN values
            df_with_nan = generate_test_data(50)
            df_with_nan.loc[df_with_nan.index[10:15], 'close'] = np.nan
            
            df_processed = strategy.calculate_indicators(df_with_nan)
            self.assert_test(not df_processed['rsi'].isna().all(), "RSI handles NaN")
            self.assert_test(not df_processed['mfi'].isna().all(), "MFI handles NaN")
            
            # Test with extreme values
            df_extreme = generate_test_data(50)
            df_extreme.loc[df_extreme.index[25], 'close'] = 999999  # Extreme spike
            
            df_processed = strategy.calculate_indicators(df_extreme)
            self.assert_test((df_processed['rsi'] <= 100).all(), "RSI bounded with extreme values")
            self.assert_test((df_processed['mfi'] <= 100).all(), "MFI bounded with extreme values")
            
            # Test with flat prices
            df_flat = generate_test_data(50)
            df_flat['close'] = 0.082  # All same price
            df_flat['high'] = 0.082
            df_flat['low'] = 0.082
            df_flat['open'] = 0.082
            
            for col in ['close', 'high', 'low', 'open']:
              df_flat.loc[:, col] = 0.082  # All same price
            
            df_processed = strategy.calculate_indicators(df_flat)
            self.assert_test(not df_processed.empty, "Processes flat prices")
            
        except Exception as e:
            self.assert_test(False, "Data processing", str(e))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n⚠️ Testing Edge Cases...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            # Test empty dataframe
            empty_df = pd.DataFrame()
            result = strategy.calculate_indicators(empty_df)
            self.assert_test(result.empty, "Handles empty dataframe")
            
            # Test single row
            single_row = generate_test_data(1)
            result = strategy.calculate_indicators(single_row)
            self.assert_test(len(result) == 1, "Handles single row")
            
            # Test very small values
            small_df = generate_test_data(50, start_price=0.000001)
            result = strategy.calculate_indicators(small_df)
            self.assert_test(not result.empty, "Handles very small prices")
            
            # Test signal generation with no position
            strategy.reset_position()
            signal = strategy.generate_signal(generate_test_data(100))
            # Should work without error
            self.assert_test(True, "Signal generation with no position")
            
        except Exception as e:
            self.assert_test(False, "Edge cases", str(e))
    
    def test_integration(self):
        """Test component integration"""
        print("\n🔗 Testing Integration...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            from core.risk_management import RiskManager
            
            strategy = RSIMFICloudStrategy()
            risk_manager = RiskManager()
            
            # Test that both use same symbol
            self.assert_test(strategy.symbol == risk_manager.symbol, "Matching symbols")
            
            # Test full workflow
            df = generate_test_data(100)
            
            # Generate signal
            signal = strategy.generate_signal(df)
            
            if signal and signal['action'] in ['BUY', 'SELL']:
                # Calculate position size
                balance = 1000
                price = signal['price']
                pos_size = risk_manager.calculate_position_size(balance, price)
                
                # Calculate stops
                if signal['action'] == 'BUY':
                    sl = risk_manager.get_stop_loss(price, 'long')
                    tp = risk_manager.get_take_profit(price, 'long')
                else:
                    sl = risk_manager.get_stop_loss(price, 'short')
                    tp = risk_manager.get_take_profit(price, 'short')
                
                self.assert_test(pos_size > 0, "Positive position size")
                self.assert_test(sl != price, "Stop loss different from entry")
                self.assert_test(tp != price, "Take profit different from entry")
            
            self.assert_test(True, "Full workflow integration")
            
        except Exception as e:
            self.assert_test(False, "Integration test", str(e))
    
    def test_performance(self):
        """Test performance with larger datasets"""
        print("\n⚡ Testing Performance...")
        
        try:
            from strategies.RSI_MFI_Cloud import RSIMFICloudStrategy
            strategy = RSIMFICloudStrategy()
            
            # Test with large dataset
            large_df = generate_test_data(1000)
            
            start_time = datetime.now()
            df_processed = strategy.calculate_indicators(large_df)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.assert_test(processing_time < 5.0, f"Processing time reasonable: {processing_time:.2f}s")
            self.assert_test(len(df_processed) == 1000, "Processed all rows")
            
            # Test signal generation performance
            start_time = datetime.now()
            for i in range(10):
                signal = strategy.generate_signal(large_df)
            signal_time = (datetime.now() - start_time).total_seconds()
            
            self.assert_test(signal_time < 2.0, f"Signal generation time reasonable: {signal_time:.2f}s")
            
        except Exception as e:
            self.assert_test(False, "Performance test", str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 ZORA Trading Bot - Comprehensive Test Suite")
        print("=" * 60)
        
        self.setup()
        
        try:
            # Run all tests
            self.test_risk_manager()
            self.test_strategy_indicators() 
            self.test_signal_generation()
            self.test_atr_risk_management()
            await self.test_telegram_notifier()
            self.test_data_processing()
            self.test_edge_cases()
            self.test_integration()
            self.test_performance()
            
            # Summary
            print("\n" + "=" * 60)
            print(f"📊 TEST RESULTS:")
            print(f"✅ Passed: {self.passed}")
            print(f"❌ Failed: {self.failed}")
            print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
            
            if self.failed == 0:
                print("🎉 ALL TESTS PASSED! System ready for deployment.")
            else:
                print("⚠️ Some tests failed. Review before deployment.")
            
        finally:
            self.cleanup()

async def main():
    """Main test runner"""
    try:
        # Install required packages check
        required_packages = ['pandas', 'numpy', 'pandas_ta']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return
        
        # Run tests
        test_suite = TestSuite()
        await test_suite.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")

if __name__ == "__main__":
    asyncio.run(main())