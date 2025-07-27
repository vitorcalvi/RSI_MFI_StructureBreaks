#!/usr/bin/env python3
"""
Streamlined Testnet Trading Test
Quick test to verify trading functionality on Bybit testnet
"""

import asyncio
import os
import sys
from datetime import datetime
from pybit.unified_trading import HTTP

# Simple color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

class TestnetTrader:
    def __init__(self):
        self.api_key = 'bTOThxe1hgVDm2iZV0'
        self.api_secret = 'BpMTdZqTUXwlR9IW6Mk5qZmLbOgViBx8Nrcx'
        self.symbol = 'SOLUSDT'
        self.session = None
        
    def connect(self):
        """Connect to Bybit testnet"""
        try:
            self.session = HTTP(
                testnet=True,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            print_success("Connected to Bybit Testnet")
            return True
        except Exception as e:
            print_error(f"Connection failed: {e}")
            return False
    
    def check_balance(self):
        """Check USDT balance"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if response['retCode'] == 0:
                for coin in response['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        balance = float(coin['walletBalance'])
                        print_success(f"USDT Balance: ${balance:.2f}")
                        return balance
            else:
                print_error(f"Balance check failed: {response['retMsg']}")
                return 0
        except Exception as e:
            print_error(f"Balance error: {e}")
            return 0
    
    def get_current_price(self):
        """Get current market price"""
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                price = float(response['result']['list'][0]['lastPrice'])
                print_info(f"Current {self.symbol} price: ${price:.4f}")
                return price
            else:
                print_error("Failed to get price")
                return None
        except Exception as e:
            print_error(f"Price error: {e}")
            return None
    
    def check_position(self):
        """Check current position"""
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                positions = response['result']['list']
                for pos in positions:
                    if float(pos['size']) > 0:
                        side = pos['side']
                        size = float(pos['size'])
                        avg_price = float(pos['avgPrice'])
                        pnl = float(pos['unrealisedPnl'])
                        print_info(f"Active {side} position: {size} contracts @ ${avg_price:.4f}, PnL: ${pnl:.2f}")
                        return {
                            'side': side,
                            'size': size,
                            'avg_price': avg_price,
                            'pnl': pnl
                        }
                print_info("No active position")
                return None
            else:
                print_error(f"Position check failed: {response['retMsg']}")
                return None
        except Exception as e:
            print_error(f"Position error: {e}")
            return None
    
    def place_order(self, side, contracts):
        """Place a market order"""
        try:
            print_info(f"Placing {side} order for {contracts} contracts...")
            
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(contracts),
                positionIdx=0  # One-way mode
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                print_success(f"{side} order placed! Order ID: {order_id}")
                return order_id
            else:
                print_error(f"Order failed: {response['retMsg']}")
                return None
        except Exception as e:
            print_error(f"Order error: {e}")
            return None
    
    def close_position(self, position):
        """Close an existing position"""
        try:
            # Opposite side to close
            close_side = "Sell" if position['side'] == "Buy" else "Buy"
            
            print_info(f"Closing {position['side']} position...")
            
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=close_side,
                orderType="Market",
                qty=str(position['size']),
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                print_success(f"Position closed! PnL: ${position['pnl']:.2f}")
                return True
            else:
                print_error(f"Close failed: {response['retMsg']}")
                return False
        except Exception as e:
            print_error(f"Close error: {e}")
            return False

async def run_test():
    """Run the streamlined test"""
    print("=" * 60)
    print("🚀 STREAMLINED TESTNET TRADING TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    trader = TestnetTrader()
    
    # Step 1: Connect
    print("STEP 1: Connecting to Bybit Testnet...")
    if not trader.connect():
        return
    
    # Step 2: Check balance
    print("\nSTEP 2: Checking balance...")
    balance = trader.check_balance()
    if balance < 10:
        print_error("Insufficient balance! Need at least $10 USDT")
        print_info("Get testnet funds from: https://testnet.bybit.com/faucet")
        return
    
    # Step 3: Get market price
    print("\nSTEP 3: Getting market data...")
    price = trader.get_current_price()
    if not price:
        return
    
    # Step 4: Check for existing position
    print("\nSTEP 4: Checking positions...")
    position = trader.check_position()
    
    # Step 5: Execute test trades
    print("\nSTEP 5: Executing test trades...")
    
    # Calculate small test position (0.5% of balance)
    position_value = balance * 0.005
    contracts = round(position_value / price, 3)
    
    print_info(f"Test position size: {contracts} contracts (${position_value:.2f})")
    
    if position:
        # Close existing position first
        print_warning("Closing existing position first...")
        if trader.close_position(position):
            await asyncio.sleep(2)
        else:
            return
    
    # Test 1: Open LONG
    print("\n📈 TEST 1: Opening LONG position...")
    order_id = trader.place_order("Buy", contracts)
    if not order_id:
        return
    
    await asyncio.sleep(3)
    
    # Check position
    position = trader.check_position()
    if not position:
        print_error("Failed to open position")
        return
    
    # Test 2: Close LONG
    print("\n📉 TEST 2: Closing LONG position...")
    if trader.close_position(position):
        await asyncio.sleep(3)
    
    # Test 3: Open SHORT
    print("\n📉 TEST 3: Opening SHORT position...")
    order_id = trader.place_order("Sell", contracts)
    if not order_id:
        return
    
    await asyncio.sleep(3)
    
    # Check position
    position = trader.check_position()
    if not position:
        print_error("Failed to open SHORT position")
        return
    
    # Test 4: Close SHORT
    print("\n📈 TEST 4: Closing SHORT position...")
    if trader.close_position(position):
        await asyncio.sleep(3)
    
    # Final balance check
    print("\nFINAL: Checking final balance...")
    final_balance = trader.check_balance()
    pnl = final_balance - balance
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print_success("All tests completed!")
    print(f"Initial Balance: ${balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Net P&L: ${pnl:+.2f}")
    print("=" * 60)

def main():
    """Main entry point"""
    print("Bybit Testnet Trading Test")
    print("This will execute real trades on testnet!")
    print()
    
    response = input("Continue with test? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print_error(f"Test failed: {e}")

if __name__ == "__main__":
    main()