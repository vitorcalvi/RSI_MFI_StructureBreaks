#!/usr/bin/env python3
"""
Automated Setup and System Check for Trading Bot
Handles installation, configuration, and testing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing Dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists('requirements.txt'):
            print("❌ requirements.txt not found")
            return False
            
        # Install dependencies
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("\n⚙️ Setting up configuration...")
    
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True
        
    env_template = """# Trading Configuration
SYMBOLS=SOL/USDT
EXCHANGE=bybit
DEMO_MODE=true

# Bybit Live Trading API (set DEMO_MODE=false to use)
BYBIT_API_KEY=your_live_api_key_here
BYBIT_API_SECRET=your_live_secret_here

# Bybit Testnet API (for demo mode)
TESTNET_BYBIT_API_KEY=your_testnet_api_key_here
TESTNET_BYBIT_API_SECRET=your_testnet_secret_here

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Notes:
# - Get Bybit testnet keys from: https://testnet.bybit.com
# - Get Bybit live keys from: https://www.bybit.com  
# - Create Telegram bot with @BotFather for notifications
# - Start with DEMO_MODE=true for paper trading
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env file with default configuration")
        print("💡 Edit .env file to add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_file_structure():
    """Verify all required files exist"""
    print("\n📁 Checking File Structure...")
    
    required_files = [
        'main.py',
        'comprehensive_test.py',
        'core/trade_engine.py',
        'core/risk_management.py',
        'core/telegram_notifier.py',
        'strategies/RSI_MFI_Cloud.py',
        'strategies/params_RSI_MFI_Cloud.json',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
            
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    else:
        print("\n✅ All required files present")
        return True

def run_comprehensive_test():
    """Run the comprehensive system test"""
    print("\n🧪 Running Comprehensive System Test...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, 'comprehensive_test.py'
        ], capture_output=False, text=True)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def show_next_steps(test_passed):
    """Show next steps based on test results"""
    print("\n" + "=" * 50)
    print("🎯 NEXT STEPS")
    print("=" * 50)
    
    if test_passed:
        print("🎉 System is ready!")
        print("\n📋 To start trading:")
        print("1. Edit .env file with your API keys (optional for demo mode)")
        print("2. Run: python main.py")
        print("\n💡 Tips:")
        print("- Start with DEMO_MODE=true for safety")
        print("- Monitor Telegram for trade alerts")
        print("- Check strategies/params_RSI_MFI_Cloud.json to adjust settings")
        
    else:
        print("⚠️  System needs attention!")
        print("\n🔧 Common fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check .env file configuration")
        print("3. Verify API keys at https://testnet.bybit.com")
        print("4. Run: python comprehensive_test.py")
        
    print("\n📚 Documentation:")
    print("- Strategy params: strategies/params_RSI_MFI_Cloud.json")
    print("- Risk settings: core/risk_management.py")
    print("- Test reports: test_report_*.json")

def main():
    """Main setup routine"""
    print("🚀 TRADING BOT SETUP & SYSTEM CHECK")
    print("=" * 50)
    print("This script will:")
    print("1. Check Python version")
    print("2. Install dependencies")
    print("3. Create configuration files")
    print("4. Verify file structure")
    print("5. Run comprehensive tests")
    print("6. Show next steps")
    print()
    
    # Setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Create Configuration", create_env_file),
        ("File Structure Check", check_file_structure)
    ]
    
    setup_success = True
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            setup_success = False
            print(f"❌ {step_name} failed")
            break
        else:
            print(f"✅ {step_name} completed")
    
    if not setup_success:
        print("\n❌ Setup failed. Fix errors above and try again.")
        return False
    
    # Run comprehensive test
    print(f"\n{'='*20} System Test {'='*20}")
    test_passed = run_comprehensive_test()
    
    # Show next steps
    show_next_steps(test_passed)
    
    return test_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)