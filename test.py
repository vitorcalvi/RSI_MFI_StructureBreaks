#!/usr/bin/env python3
"""
Debug authentication issues with Bybit API
"""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Load environment variables
load_dotenv()

print("🔍 Debugging Bybit Authentication\n")

# Check environment variables
print("1. Environment Variables:")
api_key = os.getenv('TESTNET_BYBIT_API_KEY')
api_secret = os.getenv('TESTNET_BYBIT_API_SECRET')

if api_key:
    print(f"   ✓ API Key: {api_key[:4]}...{api_key[-4:]}")
else:
    print("   ✗ API Key: Not found")

if api_secret:
    print(f"   ✓ API Secret: {api_secret[:4]}...{api_secret[-4:]}")
else:
    print("   ✗ API Secret: Not found")

print(f"\n2. Creating HTTP client with demo=True")
try:
    # Create client exactly like working code
    exchange = HTTP(demo=True, api_key=api_key, api_secret=api_secret)
    print("   ✓ Client created successfully")
except Exception as e:
    print(f"   ✗ Failed to create client: {e}")
    exit(1)

print(f"\n3. Testing public endpoint (no auth required):")
try:
    server_time = exchange.get_server_time()
    if server_time.get('retCode') == 0:
        print(f"   ✓ Server time: {server_time['result']['timeSecond']}")
    else:
        print(f"   ✗ Failed: {server_time.get('retMsg')}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print(f"\n4. Testing authenticated endpoint (wallet balance):")
try:
    balance = exchange.get_wallet_balance(accountType="UNIFIED")
    if balance.get('retCode') == 0:
        print("   ✓ Authentication successful!")
        # Print balance info
        for coin in balance['result']['list'][0].get('coin', []):
            if coin.get('coin') == 'USDT':
                print(f"   USDT Balance: {coin.get('walletBalance', 0)}")
    else:
        print(f"   ✗ Authentication failed: {balance.get('retMsg')}")
        print(f"   Error code: {balance.get('retCode')}")
        
        # Common issues
        if balance.get('retCode') == 10003:
            print("\n   💡 Invalid API key. Check your API key is correct.")
        elif balance.get('retCode') == 10004:
            print("\n   💡 Invalid signature. Check your API secret is correct.")
        elif balance.get('retCode') == 401:
            print("\n   💡 Unauthorized. Possible issues:")
            print("      - API key/secret mismatch")
            print("      - API key not enabled for testnet")
            print("      - IP restrictions on API key")
            print("      - API key permissions insufficient")
            
except Exception as e:
    print(f"   ✗ Error: {e}")

print(f"\n5. Testing position endpoint:")
try:
    positions = exchange.get_positions(category="linear", symbol="SOLUSDT")
    if positions.get('retCode') == 0:
        print("   ✓ Position check successful!")
    else:
        print(f"   ✗ Position check failed: {positions.get('retMsg')}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*50)
print("📋 Troubleshooting steps:")
print("1. Verify API keys in .env file match Bybit testnet")
print("2. Check API key permissions include 'Spot' and 'Derivatives'")
print("3. Ensure no IP restrictions on testnet API key")
print("4. Try regenerating API keys on testnet.bybit.com")
print("="*50)