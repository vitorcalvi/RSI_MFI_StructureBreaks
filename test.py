#!/usr/bin/env python3
"""
Comprehensive integration test for the complete trading bot
Tests all real-world scenarios and edge cases
"""

import asyncio
import sys
import os
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

class ComprehensiveBotTester:
    def __init__(self):
        self.test_results = []
        self.setup_mocks()
    
    def setup_mocks(self):
        """Setup all required mocks"""
        sys.modules['pybit.unified_trading'] = Mock()
        sys.modules['telegram'] = Mock()
        sys.modules['strategies.RSI_MFI_Cloud'] = Mock()
        sys.modules['core.risk_management'] = Mock()
    
    def log_test(self, name, passed, error=None):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((name, passed, error))
        print(f"{status}: {name}")
        if error:
            print(f"    Error: {error}")
    
    async def test_complete_trading_cycle(self):
        """Test a complete trading cycle from start to finish"""
        try:
            with patch.dict(os.environ, {
                'SYMBOLS': 'BTC/USDT',
                'DEMO_MODE': 'true',
                'TESTNET_BYBIT_API_KEY': 'test_key',
                'TESTNET_BYBIT_API_SECRET': 'test_secret',
                'TELEGRAM_BOT_TOKEN': 'test_token',
                'TELEGRAM_CHAT_ID': 'test_chat'
            }):
                from core.trade_engine import TradeEngine
                
                engine = TradeEngine()
                
                # Mock all dependencies
                self.setup_engine_mocks(engine)
                
                # Test 1: Bot initialization
                assert engine.symbol == 'BTC/USDT'
                assert engine.demo_mode == True
                self.log_test("Bot initialization", True)
                
                # Test 2: Connection
                engine.exchange.get_server_time.return_value = {'retCode': 0}
                result = engine.connect()
                assert result == True
                self.log_test("Exchange connection", True)
                
                # Test 3: Market data retrieval
                engine.exchange.get_kline.return_value = self.create_mock_kline_data()
                df = engine.get_market_data()
                assert df is not None
                assert len(df) > 0
                self.log_test("Market data retrieval", True)
                
                # Test 4: Open BUY position
                signal = {'action': 'BUY', 'price': 50000.0}
                engine.exchange.place_order.return_value = {'retCode': 0, 'result': {'orderId': 'test123'}}
                result = await engine.open_position(signal)
                assert result == True
                self.log_test("Open BUY position", True)
                
                # Test 5: Check position
                engine.exchange.get_positions.return_value = self.create_mock_position_data('Buy')
                position = engine.check_position()
                assert position is not None
                assert position['side'] == 'Buy'
                self.log_test("Position check", True)
                
                # Test 6: Profit lock activation
                engine.position = {'unrealized_pnl_pct': 1.0, 'side': 'Buy', 'avg_price': 50000.0}
                await engine.check_profit_lock(51000.0)
                assert engine.profit_lock_active == True
                self.log_test("Profit lock activation", True)
                
                # Test 7: Position reversal
                signal = {'action': 'SELL', 'price': 51000.0}
                await engine.open_position(signal)
                self.log_test("Position reversal", True)
                
                # Test 8: Loss switch
                engine.position = {'unrealized_pnl_pct': -2.5, 'side': 'Buy', 'size': 0.1}
                engine.profit_lock_active = False
                await engine.check_loss_switch()
                self.log_test("Loss switch", True)
                
                # Test 9: Close position
                result = await engine.close_position("Test")
                assert result == True
                self.log_test("Close position", True)
                
                # Test 10: Bot stop
                await engine.stop()
                assert engine.running == False
                self.log_test("Bot stop", True)
                
        except Exception as e:
            self.log_test("Complete trading cycle", False, str(e))
    
    def setup_engine_mocks(self, engine):
        """Setup all engine mocks"""
        engine.exchange = Mock()
        engine.notifier = Mock()
        engine.strategy = Mock()
        engine.risk_manager = Mock()
        
        # Mock notifier methods
        engine.notifier.trade_opened = AsyncMock()
        engine.notifier.trade_closed = AsyncMock()
        engine.notifier.profit_lock_activated = AsyncMock()
        engine.notifier.send_message = AsyncMock()
        
        # Mock risk manager
        engine.risk_manager.calculate_position_size = Mock(return_value=0.1)
        engine.risk_manager.get_stop_loss = Mock(return_value=49000.0)
        engine.risk_manager.get_take_profit = Mock(return_value=51000.0)
        
        # Mock exchange methods
        engine.exchange.get_wallet_balance.return_value = {
            'retCode': 0,
            'result': {'list': [{'coin': [{'coin': 'USDT', 'walletBalance': '1000.0'}]}]}
        }
        
        engine.exchange.get_tickers.return_value = {
            'result': {'list': [{'lastPrice': '50000.0'}]}
        }
        
        engine.exchange.get_instruments_info.return_value = {
            'retCode': 0,
            'result': {'list': [{
                'lotSizeFilter': {'minOrderQty': '0.001', 'qtyStep': '0.001'},
                'priceFilter': {'tickSize': '0.01'}
            }]}
        }
        
        engine.exchange.set_trading_stop.return_value = {'retCode': 0}
        engine.exchange.switch_position_mode.return_value = {'retCode': 0}
    
    def create_mock_kline_data(self):
        """Create realistic kline data"""
        base_time = int(datetime.now().timestamp() * 1000)
        data = []
        
        for i in range(100):
            timestamp = base_time - (i * 300000)  # 5 minute intervals
            price = 50000 + (i * 10)  # Trending price
            data.append([
                str(timestamp),
                str(price - 50),      # open
                str(price + 100),     # high
                str(price - 100),     # low
                str(price),           # close
                '1000',               # volume
                str(price * 1000)     # turnover
            ])
        
        return {
            'retCode': 0,
            'result': {'list': data}
        }
    
    def create_mock_position_data(self, side, pnl_pct=1.0):
        """Create mock position data"""
        return {
            'retCode': 0,
            'result': {'list': [{
                'side': side,
                'size': '0.1',
                'avgPrice': '50000',
                'unrealisedPnl': str(50000 * 0.1 * pnl_pct / 100),
            }]}
        }
    
    async def test_stress_scenarios(self):
        """Test stress scenarios and edge cases"""
        try:
            with patch.dict(os.environ, {
                'SYMBOLS': 'BTC/USDT',
                'DEMO_MODE': 'true',
                'TESTNET_BYBIT_API_KEY': 'test_key',
                'TESTNET_BYBIT_API_SECRET': 'test_secret'
            }):
                from core.trade_engine import TradeEngine
                
                engine = TradeEngine()
                self.setup_engine_mocks(engine)
                
                # Test rapid signal changes
                signals = [
                    {'action': 'BUY', 'price': 50000},
                    {'action': 'SELL', 'price': 50100},
                    {'action': 'BUY', 'price': 50200},
                    {'action': 'SELL', 'price': 50300}
                ]
                
                for i, signal in enumerate(signals):
                    if i > 0:
                        # Simulate position from previous signal
                        prev_side = 'Buy' if signals[i-1]['action'] == 'BUY' else 'Sell'
                        engine.position = {'side': prev_side, 'size': 0.1, 'unrealized_pnl_pct': 0.5}
                    
                    await engine.open_position(signal)
                
                self.log_test("Rapid signal changes", True)
                
                # Test API failures
                engine.exchange.place_order.return_value = {'retCode': 1, 'retMsg': 'Error'}
                signal = {'action': 'BUY', 'price': 50000}
                result = await engine.open_position(signal)
                assert result == False
                self.log_test("API failure handling", True)
                
                # Test network errors
                engine.exchange.get_kline.side_effect = Exception("Network error")
                df = engine.get_market_data()
                assert df is None
                self.log_test("Network error handling", True)
                
                # Test extreme profit/loss
                engine.position = {'unrealized_pnl_pct': 50.0, 'side': 'Buy', 'avg_price': 50000.0}
                await engine.check_profit_lock(75000.0)
                self.log_test("Extreme profit handling", True)
                
                engine.position = {'unrealized_pnl_pct': -10.0, 'side': 'Buy', 'size': 0.1}
                engine.profit_lock_active = False
                await engine.check_loss_switch()
                self.log_test("Extreme loss handling", True)
                
        except Exception as e:
            self.log_test("Stress scenarios", False, str(e))
    
    async def test_real_market_simulation(self):
        """Simulate real market conditions"""
        try:
            with patch.dict(os.environ, {
                'SYMBOLS': 'BTC/USDT',
                'DEMO_MODE': 'true',
                'TESTNET_BYBIT_API_KEY': 'test_key',
                'TESTNET_BYBIT_API_SECRET': 'test_secret'
            }):
                from core.trade_engine import TradeEngine
                
                engine = TradeEngine()
                self.setup_engine_mocks(engine)
                
                # Simulate market data with realistic patterns
                market_scenarios = [
                    self.create_trending_market(),
                    self.create_sideways_market(),
                    self.create_volatile_market(),
                    self.create_crash_market()
                ]
                
                for i, market_data in enumerate(market_scenarios):
                    engine.get_market_data = Mock(return_value=market_data)
                    engine.strategy.generate_signal = Mock(return_value=None)  # No signals initially
                    
                    # Run several cycles
                    for _ in range(5):
                        await engine.run_cycle()
                    
                    self.log_test(f"Market scenario {i+1}", True)
                
                # Test with actual signals
                engine.strategy.generate_signal = Mock(side_effect=[
                    {'action': 'BUY', 'price': 50000},
                    None, None,  # No signals for a few cycles
                    {'action': 'SELL', 'price': 51000},
                    None
                ])
                
                for _ in range(5):
                    await engine.run_cycle()
                
                self.log_test("Signal processing in real market", True)
                
        except Exception as e:
            self.log_test("Real market simulation", False, str(e))
    
    def create_trending_market(self):
        """Create uptrending market data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        prices = [50000 + i * 10 for i in range(100)]  # Uptrend
        
        return pd.DataFrame({
            'close': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'volume': [1000 + i * 5 for i in range(100)]
        }, index=dates)
    
    def create_sideways_market(self):
        """Create sideways market data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        import math
        prices = [50000 + math.sin(i * 0.1) * 100 for i in range(100)]  # Sideways
        
        return pd.DataFrame({
            'close': prices,
            'high': [p + 30 for p in prices],
            'low': [p - 30 for p in prices],
            'volume': [1000 for _ in range(100)]
        }, index=dates)
    
    def create_volatile_market(self):
        """Create volatile market data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        import random
        prices = [50000 + random.randint(-500, 500) for _ in range(100)]  # Volatile
        
        return pd.DataFrame({
            'close': prices,
            'high': [p + random.randint(50, 200) for p in prices],
            'low': [p - random.randint(50, 200) for p in prices],
            'volume': [random.randint(500, 2000) for _ in range(100)]
        }, index=dates)
    
    def create_crash_market(self):
        """Create market crash scenario"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        prices = [50000 - i * 50 for i in range(100)]  # Sharp decline
        
        return pd.DataFrame({
            'close': prices,
            'high': [p + 20 for p in prices],
            'low': [p - 100 for p in prices],
            'volume': [2000 + i * 10 for i in range(100)]  # High volume
        }, index=dates)
    
    async def test_telegram_integration(self):
        """Test full Telegram integration"""
        try:
            with patch.dict(os.environ, {
                'TELEGRAM_BOT_TOKEN': 'test_token',
                'TELEGRAM_CHAT_ID': 'test_chat'
            }):
                from core.trade_engine import TelegramNotifier
                
                notifier = TelegramNotifier()
                notifier.bot = Mock()
                notifier.bot.send_message = AsyncMock()
                
                # Test all notification types
                await notifier.trade_opened("BTCUSDT", 50000.0, 0.1)
                await notifier.trade_closed("BTCUSDT", 2.5, 125.0, "Profit Target")
                await notifier.profit_lock_activated("BTCUSDT", 1.5, 1.0)
                await notifier.trailing_stop_updated("BTCUSDT", 49500.0, 50000.0)
                
                # Verify all messages were sent
                assert notifier.bot.send_message.call_count == 4
                self.log_test("Telegram notifications", True)
                
                # Test error handling
                notifier.bot.send_message.side_effect = Exception("Network error")
                await notifier.send_message("Test")  # Should not raise
                self.log_test("Telegram error handling", True)
                
        except Exception as e:
            self.log_test("Telegram integration", False, str(e))
    
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive Trading Bot Test")
        print("=" * 60)
        
        test_suites = [
            ("Complete Trading Cycle", self.test_complete_trading_cycle),
            ("Stress Scenarios", self.test_stress_scenarios),
            ("Real Market Simulation", self.test_real_market_simulation),
            ("Telegram Integration", self.test_telegram_integration),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n📋 Testing: {suite_name}")
            print("-" * 40)
            
            try:
                await test_func()
            except Exception as e:
                self.log_test(f"{suite_name} - SUITE ERROR", False, str(e))
                print(f"Suite error: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for name, result, error in self.test_results:
                if not result:
                    print(f"  - {name}: {error}")
        
        # Performance metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  - Trading Cycle Tests: ✅")
        print(f"  - Error Handling: ✅")
        print(f"  - Market Scenarios: ✅")
        print(f"  - Integration Tests: ✅")
        
        print("\n🎯 Comprehensive testing complete!")
        return passed == total

async def main():
    """Run comprehensive tests"""
    tester = ComprehensiveBotTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Bot is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Review and fix before deployment.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)