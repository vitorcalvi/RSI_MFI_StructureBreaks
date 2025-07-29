#!/usr/bin/env python3
"""
Unified Trading Tool for Jesse Bot
Usage:
    # Basic market orders
    python trade_bybit.py buy/sell ETH/USDT                 # Market buy/sell with defaults (0.4% SL, 1:3 RR)
    
    # Limit orders
    python trade_bybit.py buy ETH/USDT --limit 3500         # Limit buy at $3,500
    
    # Custom stop loss
    python trade_bybit.py buy SOL/USDT --stop 0.5          # 0.5% stop loss (TP auto 1.5% with 1:3 RR)
    python trade_bybit.py buy SOL/USDT --sl 1.0            # 1% stop loss (TP auto 3% with 1:3 RR)
    
    # Custom risk/reward ratio
    python trade_bybit.py buy ETH/USDT --rr 5              # 1:5 risk/reward (0.4% SL, 2% TP)
    python trade_bybit.py sell BTC/USDT --stop 0.3 --rr 4  # 0.3% SL with 1:4 RR = 1.2% TP
    
    # Custom take profit (overrides RR)
    python trade_bybit.py buy SOL/USDT --tp 2.5            # 2.5% take profit (RR auto-calculated)
    python trade_bybit.py buy ETH/USDT --sl 0.5 --tp 3     # 0.5% SL, 3% TP = 1:6 RR
    
    # Trailing stop
    python trade_bybit.py buy BTC/USDT --trail 0.2         # 0.2% trailing (default 0.1%)
    python trade_bybit.py sell ETH/USDT --trail 0.5        # 0.5% trailing stop
    
    # Combined parameters
    python trade_bybit.py buy ETH/USDT --limit 3500 --stop 0.6 --rr 4      # Limit with custom SL/RR
    python trade_bybit.py sell SOL/USDT --sl 0.8 --tp 4 --trail 0.3        # Custom SL/TP/Trail
    python trade_bybit.py buy BTC/USDT --market --stop 1 --trail 0.5       # Explicit market order
    
    # Real money trading (requires confirmation)
    python trade_bybit.py buy ETH/USDT --real              # Real market buy
    python trade_bybit.py sell BTC/USDT --limit 95000 --real  # Real limit sell
"""

import argparse
import sys
import os
import time
from pybit.unified_trading import HTTP
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from threading import Thread
import csv

# Load environment variables
load_dotenv()

# Color support
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    G, R, Y, B, BOLD, RST = Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.CYAN, Style.BRIGHT, Style.RESET_ALL
except ImportError:
    G = R = Y = B = BOLD = RST = ""


class Config:
    """Trading configuration"""
    DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'
    
    # API credentials
    if DEMO_MODE:
        API_KEY = os.getenv('TESTNET_BYBIT_API_KEY')
        API_SECRET = os.getenv('TESTNET_BYBIT_API_SECRET')
    else:
        API_KEY = os.getenv('LIVE_BYBIT_API_KEY')
        API_SECRET = os.getenv('LIVE_BYBIT_API_SECRET')
    
    # Defaults
    POSITION_SIZE = float(os.getenv('DEFAULT_POSITION_SIZE_PCT', '0.05'))
    STOP_LOSS = float(os.getenv('STOP_LOSS_PCT', '0.004'))
    RISK_RATIO = 2.0  # Changed from 3.0 to 2.0
    TRAILING = 0.01   # Changed from 0.001 to 0.01 (1%)
    
    # Data files
    DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
    TRADES_FILE = DATA_DIR / 'trades.csv'
    
    @classmethod
    def init(cls):
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError("API credentials not configured")
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not cls.TRADES_FILE.exists():
            with open(cls.TRADES_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'order_type', 'qty', 
                    'entry_price', 'actual_entry', 'stop_loss', 'take_profit', 
                    'risk_reward', 'trailing_pct', 'risk_usdt', 'reward_usdt',
                    'net_risk', 'net_reward', 'net_rr', 'balance', 
                    'position_size_pct', 'atr', 'mode'
                ])


def get_exchange():
    """Initialize exchange connection"""
    return HTTP(demo=Config.DEMO_MODE, api_key=Config.API_KEY, api_secret=Config.API_SECRET)


def get_balance(exchange):
    """Get USDT balance"""
    resp = exchange.get_wallet_balance(accountType="UNIFIED")
    if resp.get('retCode') == 0:
        for coin in resp['result']['list'][0].get('coin', []):
            if coin.get('coin') == 'USDT':
                return float(coin.get('walletBalance', 0))
    return 0


def get_symbol_info(exchange, symbol):
    """Get symbol trading rules"""
    resp = exchange.get_instruments_info(category="linear", symbol=symbol.replace('/', ''))
    if resp.get('retCode') == 0 and resp['result']['list']:
        info = resp['result']['list'][0]
        return {
            'min_qty': float(info['lotSizeFilter']['minOrderQty']),
            'qty_step': float(info['lotSizeFilter']['qtyStep']),
            'tick_size': float(info['priceFilter']['tickSize'])
        }
    return None


def calculate_atr(exchange, symbol, period=14, interval="15"):
    """Calculate ATR (Average True Range)"""
    try:
        # Get kline data
        resp = exchange.get_kline(
            category="linear",
            symbol=symbol.replace('/', ''),
            interval=interval,
            limit=period + 1
        )
        
        if resp.get('retCode') != 0 or not resp.get('result', {}).get('list'):
            return None
            
        klines = resp['result']['list']
        
        # Calculate True Range for each candle
        true_ranges = []
        for i in range(len(klines) - 1):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i + 1][4])
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Calculate ATR (average of true ranges)
        if true_ranges:
            atr = sum(true_ranges) / len(true_ranges)
            return atr
            
    except Exception as e:
        print(f"{R}Error calculating ATR: {e}{RST}")
    
    return None


def format_qty(info, raw_qty):
    """Format quantity according to exchange requirements"""
    step = info['qty_step']
    qty = float(int(raw_qty / step) * step)
    qty = max(qty, info['min_qty'])
    
    # Determine decimal places
    step_str = f"{step:g}"
    decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
    return f"{qty:.{decimals}f}" if decimals else str(int(qty))


def format_price(info, price):
    """Format price according to tick size"""
    tick = info['tick_size']
    price = round(price / tick) * tick
    
    # Determine decimal places based on tick size
    if tick >= 1:
        decimals = 0
    elif tick >= 0.1:
        decimals = 1
    elif tick >= 0.01:
        decimals = 2
    elif tick >= 0.001:
        decimals = 3
    elif tick >= 0.0001:
        decimals = 4
    elif tick >= 0.00001:
        decimals = 5
    else:
        decimals = 6
    
    return f"{price:.{decimals}f}"


def get_position_size_pct(symbol, atr_pct):
    """Calculate position size based on asset category and volatility"""
    base_asset = symbol.split('/')[0].upper()
    
    # Base position sizes by asset category
    if base_asset in ['BTC', 'ETH']:
        base_size = 0.05  # 5% for major assets
    elif base_asset in ['SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT']:
        base_size = 0.035  # 3.5% for large caps
    elif base_asset in ['MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'ARB', 'OP']:
        base_size = 0.025  # 2.5% for mid caps
    else:
        base_size = 0.015  # 1.5% for small caps and others
    
    # Adjust based on volatility (ATR as % of price)
    if atr_pct > 0.03:  # > 3% ATR
        size_multiplier = 0.5  # Reduce by 50%
    elif atr_pct > 0.02:  # > 2% ATR
        size_multiplier = 0.7  # Reduce by 30%
    elif atr_pct > 0.01:  # > 1% ATR
        size_multiplier = 0.85  # Reduce by 15%
    else:
        size_multiplier = 1.0  # No reduction
    
    final_size = base_size * size_multiplier
    
    # Cap maximum size at 5%
    return min(final_size, 0.05)


def set_trading_stops(exchange, symbol, linear, info, sl_price=None, tp_price=None, trail_pct=None, entry_price=None):
    """Set trading stops using Bybit API"""
    # First set SL/TP without trailing
    if sl_price or tp_price:
        params = {
            "category": "linear",
            "symbol": linear,
            "positionIdx": 0
        }
        
        if sl_price:
            params["stopLoss"] = format_price(info, sl_price)
            params["slTriggerBy"] = "LastPrice"
        
        if tp_price:
            params["takeProfit"] = format_price(info, tp_price)
            params["tpTriggerBy"] = "LastPrice"
        
        print(f"{Y}📤 Setting SL/TP with params: {params}{RST}")
        
        result = exchange.set_trading_stop(**params)
        
        if result.get('retCode') != 0:
            print(f"{R}❌ Failed to set SL/TP: {result.get('retMsg')}{RST}")
            return False
        else:
            print(f"{G}✅ SL/TP set successfully{RST}")
            if sl_price:
                print(f"   Stop Loss: ${format_price(info, sl_price)}")
            if tp_price:
                print(f"   Take Profit: ${format_price(info, tp_price)}")
    
    # Set trailing stop separately if provided
    if trail_pct and entry_price:
        # Calculate trailing amount as absolute price
        trail_amount = entry_price * (trail_pct / 100)
        trail_amount_formatted = format_price(info, trail_amount)
        
        trail_params = {
            "category": "linear",
            "symbol": linear,
            "positionIdx": 0,
            "trailingStop": trail_amount_formatted
        }
        
        print(f"{Y}📤 Setting trailing stop: {trail_pct:.1f}% = ${trail_amount_formatted}{RST}")
        
        trail_result = exchange.set_trading_stop(**trail_params)
        
        if trail_result.get('retCode') == 0:
            print(f"{G}✅ Trailing stop set: {trail_pct:.1f}%{RST}")
        else:
            print(f"{R}❌ Failed to set trailing stop: {trail_result.get('retMsg')}{RST}")
            print(f"{Y}💡 Manual trailing available - use Bybit interface{RST}")
    
    return True


def wait_for_limit_fill(exchange, linear, order_id, limit_price, side, timeout=3600):
    """Wait for limit order to fill with time-based price adjustments"""
    print(f"{Y}⏳ Waiting for limit order to fill at ${limit_price:.4f}...{RST}")
    print(f"{Y}📊 Time-based price adjustments enabled (every 60s){RST}")
    print(f"{Y}⏰ Monitoring for {timeout//60} minutes (Ctrl+C to cancel){RST}")
    
    start = time.time()
    last_check = 0
    current_limit = limit_price
    adjustment_count = 0
    adjustment_interval = 60  # Adjust price every 60 seconds
    max_adjustments = 10  # Allow up to 10 adjustments (0.5% total movement)
    
    # Price adjustment amounts (in basis points)
    if side == "Buy":
        adjustment_bps = 5  # Increase limit by 0.05% each time
    else:
        adjustment_bps = -5  # Decrease limit by 0.05% each time
    
    try:
        while (time.time() - start) < timeout:
            # Check order status
            try:
                orders = exchange.get_open_orders(category="linear", symbol=linear, orderId=order_id)
                
                # Check if order exists
                if orders.get('retCode') == 0:
                    if not orders['result']['list']:
                        # Order not in open orders - check if it was filled
                        print(f"{Y}🔍 Order not in open orders, checking order history...{RST}")
                        
                        # Get order history to confirm if filled
                        history = exchange.get_order_history(
                            category="linear",
                            symbol=linear,
                            orderId=order_id,
                            limit=1
                        )
                        
                        if history.get('retCode') == 0 and history['result']['list']:
                            order_status = history['result']['list'][0]['orderStatus']
                            if order_status == 'Filled':
                                avg_price = float(history['result']['list'][0]['avgPrice'])
                                print(f"{G}✅ Limit order filled at ${avg_price:.4f}!{RST}")
                                return True, avg_price
                            elif order_status == 'Cancelled':
                                print(f"{R}❌ Order was cancelled{RST}")
                                return False, None
                        
                        # If we can't find the order anywhere, it might have been filled
                        print(f"{G}✅ Order completed (assumed filled){RST}")
                        return True, current_limit
                elif orders.get('retCode') == 110001:
                    # Order doesn't exist - likely filled or expired
                    print(f"{Y}Order no longer exists - checking if filled...{RST}")
                    return True, current_limit
                    
            except Exception as e:
                print(f"{Y}Warning: Error checking order status: {e}{RST}")
                # Continue monitoring instead of failing
            
            # Get current market price
            ticker = exchange.get_tickers(category="linear", symbol=linear)
            if ticker.get('retCode') == 0:
                current_market = float(ticker['result']['list'][0]['lastPrice'])
                
                # Show progress every 30 seconds
                elapsed_seconds = int(time.time() - start)
                if elapsed_seconds % 30 == 0 and elapsed_seconds != last_check:
                    elapsed_mins = elapsed_seconds // 60
                    elapsed_secs = elapsed_seconds % 60
                    spread = abs(current_market - current_limit) / current_limit * 100
                    print(f"{Y}   [{elapsed_mins}m {elapsed_secs}s] Market: ${current_market:.4f}, Limit: ${current_limit:.4f} (spread: {spread:.2f}%){RST}")
                    last_check = elapsed_seconds
                
                # Time-based price adjustment
                if elapsed_seconds >= (adjustment_count + 1) * adjustment_interval and adjustment_count < max_adjustments:
                    adjustment_count += 1
                    
                    # Calculate new limit price
                    new_limit = current_limit * (1 + adjustment_bps / 10000)
                    
                    # Get symbol info for price formatting
                    info_resp = exchange.get_instruments_info(category="linear", symbol=linear)
                    if info_resp.get('retCode') == 0 and info_resp['result']['list']:
                        tick_size = float(info_resp['result']['list'][0]['priceFilter']['tickSize'])
                        new_limit = round(new_limit / tick_size) * tick_size
                    
                    print(f"{Y}📈 Adjusting limit price: ${current_limit:.4f} → ${new_limit:.4f} (adjustment #{adjustment_count}/{max_adjustments}){RST}")
                    
                    # Try to cancel current order
                    try:
                        cancel_result = exchange.cancel_order(
                            category="linear",
                            symbol=linear,
                            orderId=order_id
                        )
                        
                        if cancel_result.get('retCode') == 0:
                            # Get order quantity from original order
                            if orders.get('result', {}).get('list') and len(orders['result']['list']) > 0:
                                qty = orders['result']['list'][0]['qty']
                                
                                # Place new limit order at adjusted price
                                new_order = exchange.place_order(
                                    category="linear",
                                    symbol=linear,
                                    side=side,
                                    orderType="Limit",
                                    qty=qty,
                                    price=f"{new_limit:.4f}",
                                    timeInForce="GTC"
                                )
                                
                                if new_order.get('retCode') == 0:
                                    order_id = new_order['result']['orderId']
                                    current_limit = new_limit
                                    print(f"{G}✅ New limit order placed at ${new_limit:.4f}{RST}")
                                else:
                                    print(f"{R}❌ Failed to place new order: {new_order.get('retMsg')}{RST}")
                                    return False, None
                        elif cancel_result.get('retCode') == 110001:
                            # Order doesn't exist - might have been filled
                            print(f"{Y}Order no longer exists for adjustment - may have been filled{RST}")
                            return True, current_limit
                    except Exception as e:
                        print(f"{Y}Could not adjust order: {e}{RST}")
                
                # Emergency market conversion if spread becomes too large (>2%)
                spread_pct = abs(current_market - current_limit) / current_limit
                if (side == "Buy" and current_market > current_limit * 1.02) or \
                   (side == "Sell" and current_market < current_limit * 0.98):
                    print(f"{Y}⚡ Price moved >2% from limit! Consider manual intervention.{RST}")
                    print(f"{Y}   Current spread: {spread_pct*100:.2f}%{RST}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n{Y}⏸️  Monitoring interrupted by user{RST}")
        
        # Try to cancel the order
        try:
            cancel_result = exchange.cancel_order(
                category="linear",
                symbol=linear,
                orderId=order_id
            )
            if cancel_result.get('retCode') == 0:
                print(f"{G}✅ Order cancelled successfully{RST}")
            elif cancel_result.get('retCode') == 110001:
                print(f"{Y}Order no longer exists (may have been filled){RST}")
            else:
                print(f"{R}❌ Failed to cancel: {cancel_result.get('retMsg')}{RST}")
        except:
            print(f"{Y}Could not cancel order - please check manually{RST}")
        
        return False, None
    
    # Timeout reached
    print(f"{Y}⏱️  Timeout reached after {timeout//60} minutes{RST}")
    try:
        cancel_result = exchange.cancel_order(
            category="linear",
            symbol=linear,
            orderId=order_id
        )
        if cancel_result.get('retCode') == 0:
            print(f"{Y}Order cancelled due to timeout{RST}")
    except:
        pass
    
    return False, None


def log_trade_to_csv(trade_data):
    """Log trade details to CSV file"""
    with open(Config.TRADES_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data['side'],
            trade_data['order_type'],
            trade_data['qty'],
            trade_data['entry_price'],
            trade_data['actual_entry'],
            trade_data['stop_loss'],
            trade_data['take_profit'],
            trade_data['risk_reward'],
            trade_data['trailing_pct'],
            trade_data['risk_usdt'],
            trade_data['reward_usdt'],
            trade_data['net_risk'],
            trade_data['net_reward'],
            trade_data['net_rr'],
            trade_data['balance'],
            trade_data['position_size_pct'],
            trade_data['atr'],
            trade_data['mode']
        ])


def place_trade(direction, symbol, stop_pct=None, rr=None, tp_pct=None, trail_pct=None, order_type="Market", limit_price=None):
    """Execute trade with specified parameters"""
    
    # Initialize
    Config.init()
    exchange = get_exchange()
    
    # Test connection
    test = exchange.get_server_time()
    if test.get('retCode') != 0:
        print(f"{R}❌ Connection failed: {test.get('retMsg')}{RST}")
        return
    
    # Get symbol info
    linear = symbol.replace('/', '')
    info = get_symbol_info(exchange, symbol)
    if not info:
        print(f"{R}❌ Invalid symbol: {symbol}{RST}")
        return
    
    # Get current price
    ticker = exchange.get_tickers(category="linear", symbol=linear)
    if ticker.get('retCode') != 0:
        print(f"{R}❌ Failed to get price{RST}")
        return
    current = float(ticker['result']['list'][0]['lastPrice'])
    
    # Calculate ATR for trailing stop and position sizing
    atr = calculate_atr(exchange, symbol)
    if atr and atr > 0:
        atr_pct = atr / current
        default_trail_pct = atr_pct * 100  # Convert to percentage
        print(f"{B}📊 ATR: ${atr:.6f} ({atr_pct*100:.2f}% of price){RST}")
    else:
        atr = 0
        atr_pct = 0.025  # Default 2.5% if ATR calculation fails
        default_trail_pct = Config.TRAILING * 100  # Convert to percentage
        print(f"{B}📊 ATR: ${atr:.6f} ({atr_pct*100:.2f}% of price){RST}")
    
    # Calculate dynamic position size based on asset and volatility
    position_size_pct = get_position_size_pct(symbol, atr_pct)
    balance = get_balance(exchange)
    size_usdt = balance * position_size_pct
    qty = format_qty(info, size_usdt / current)
    
    # Show position sizing logic
    base_asset = symbol.split('/')[0].upper()
    if base_asset in ['BTC', 'ETH']:
        asset_category = "Major"
    elif base_asset in ['SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT']:
        asset_category = "Large Cap"
    elif base_asset in ['MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'ARB', 'OP']:
        asset_category = "Mid Cap"
    else:
        asset_category = "Small Cap"
    
    print(f"{Y}📊 Position Sizing: {asset_category} ({base_asset}) × {atr_pct*100:.1f}% volatility = {position_size_pct*100:.1f}% size{RST}")
    
    # Use defaults
    stop_pct = stop_pct or Config.STOP_LOSS
    trail_pct = trail_pct or default_trail_pct
    
    # Calculate TP
    if tp_pct is None:
        rr = rr or Config.RISK_RATIO
        tp_pct = stop_pct * rr
    else:
        rr = tp_pct / stop_pct
    
    # Setup
    side = "Buy" if direction == 'long' else "Sell"
    entry = limit_price if order_type == "Limit" else current
    
    # Calculate levels
    if direction == 'long':
        sl = entry * (1 - stop_pct)
        tp = entry * (1 + tp_pct)
        be_trigger = entry * (1 + stop_pct)  # Break-even = Stop Loss distance
        sl_sign = "-"
        tp_sign = "+"
        be_sign = "+"
    else:
        sl = entry * (1 + stop_pct)
        tp = entry * (1 - tp_pct)
        be_trigger = entry * (1 - stop_pct)  # Break-even = Stop Loss distance
        sl_sign = "+"
        tp_sign = "-"
        be_sign = "-"
    
    # Risk/Reward in USDT
    risk_usdt = abs(entry - sl) * float(qty)
    reward_usdt = abs(tp - entry) * float(qty)
    
    # Trading fees (0.11% round-trip = 0.055% each way)
    fee_rate = 0.00055  # 0.055% per trade
    entry_fee = size_usdt * fee_rate
    
    # For exit fees, position size changes based on P&L
    # FIXED: Correct calculation for shorts
    if direction == 'long':
        exit_fee_at_sl = (size_usdt - risk_usdt) * fee_rate  # Size decreases when losing
        exit_fee_at_tp = (size_usdt + reward_usdt) * fee_rate  # Size increases when winning
    else:
        # For SHORTS: Logic is OPPOSITE
        exit_fee_at_sl = (size_usdt + risk_usdt) * fee_rate  # Size INCREASES when losing on short
        exit_fee_at_tp = (size_usdt - reward_usdt) * fee_rate  # Size DECREASES when winning on short
    
    # Net P&L including fees
    fees_on_loss = entry_fee + exit_fee_at_sl
    fees_on_win = entry_fee + exit_fee_at_tp
    net_loss = risk_usdt + fees_on_loss
    net_profit = reward_usdt - fees_on_win
    net_rr = net_profit / net_loss
    
    # Display with proper formatting
    print(f"\n{'━' * 60}")
    print(f"{BOLD}🎯 {side.upper()} {symbol}{RST} @ ${format_price(info, entry)}")
    print(f"Type: {order_type.upper()} ORDER")
    print(f"Mode: {'Demo' if Config.DEMO_MODE else f'{R}{BOLD}🔴 LIVE{RST}'}")
    print(f"{'━' * 60}\n")
    
    print(f"{BOLD}💵 RISK FIRST:{RST}")
    print(f"   {R}• MAX LOSS: ${net_loss:,.2f} ({(net_loss/balance)*100:.2f}% of balance) ← YOUR ONLY CONCERN{RST}")
    print(f"   • Profit Target: ${net_profit:,.2f} (bonus if hit)")
    
    # Calculate most likely profit (at break-even trigger)
    be_distance = abs(be_trigger - entry)
    be_profit_gross = be_distance * float(qty)
    # Estimate fees for partial exit at BE
    be_exit_fee = (size_usdt + be_profit_gross) * fee_rate if direction == 'long' else (size_usdt - be_profit_gross) * fee_rate
    be_profit_net = be_profit_gross - entry_fee - be_exit_fee
    be_profit_multiplier = be_profit_net / net_loss
    
    print(f"   • Most Likely Profit: ${be_profit_net:,.2f} ({be_profit_multiplier:.2f}x risk) at BE trigger")
    
    fee_impact = ((rr - net_rr) / rr) * 100
    print(f"   • Net R:R: 1:{net_rr:.2f} (after {fee_impact:.0f}% fee impact)")
    
    print(f"\n{BOLD}💰 Position:{RST}")
    print(f"   Balance: ${balance:,.2f}")
    print(f"   Size: {position_size_pct*100:.1f}% = ${size_usdt:,.2f} ({qty} {symbol.split('/')[0]})")
    
    print(f"\n{BOLD}📊 Levels:{RST}")
    print(f"   Entry: ${format_price(info, entry)}")
    print(f"   Stop Loss: ${format_price(info, sl)} ({sl_sign}{stop_pct*100:.1f}%)")
    print(f"   Take Profit: ${format_price(info, tp)} ({tp_sign}{tp_pct*100:.1f}%)")
    print(f"   Break-Even Trigger: ${format_price(info, be_trigger)} ({be_sign}{stop_pct*100:.1f}%)")
    print(f"   Trailing: {trail_pct:.2f}% (Bybit native)")
    
    win_rate_needed = 100 / (1 + net_rr)
    print(f"\n{Y}⚠️  RISK CHECK:{RST}")
    print(f"   • Can you afford to lose ${net_loss:,.2f}? YES → Proceed")
    print(f"   • Is {(net_loss/balance)*100:.2f}% risk per trade your plan? YES → Proceed")
    print(f"   • Win Rate needed for profit: >{win_rate_needed:.0f}% (with 1:{net_rr:.2f} RR)")
    
    # Win rate reference table - using ACTUAL trade parameters
    print(f"\n{BOLD}📈 Win Rate Reference (with 0.11% round-trip fees):{RST}")
    print(f"   ┌─────────────┬──────────────────────┐")
    print(f"   │ Target/Stop │ Break-Even Win Rate  │")
    print(f"   ├─────────────┼──────────────────────┤")
    
    # Calculate win rates based on ACTUAL stop loss percentage
    actual_stop = stop_pct
    
    # For 1:1 R:R
    reward_1_1 = actual_stop * 1
    rr_1_1_gross = reward_1_1 - fee_rate * 2
    rr_1_1_net = rr_1_1_gross / (actual_stop + fee_rate * 2)
    wr_1_1 = 100 / (1 + rr_1_1_net)
    
    # For 1:1.5 R:R
    reward_1_15 = actual_stop * 1.5
    rr_1_15_gross = reward_1_15 - fee_rate * 2
    rr_1_15_net = rr_1_15_gross / (actual_stop + fee_rate * 2)
    wr_1_15 = 100 / (1 + rr_1_15_net)
    
    # For 1:2 R:R
    reward_1_2 = actual_stop * 2
    rr_1_2_gross = reward_1_2 - fee_rate * 2
    rr_1_2_net = rr_1_2_gross / (actual_stop + fee_rate * 2)
    wr_1_2 = 100 / (1 + rr_1_2_net)
    
    # For 1:3 R:R
    reward_1_3 = actual_stop * 3
    rr_1_3_gross = reward_1_3 - fee_rate * 2
    rr_1_3_net = rr_1_3_gross / (actual_stop + fee_rate * 2)
    wr_1_3 = 100 / (1 + rr_1_3_net)
    
    print(f"   │ 1:1 ({actual_stop*100:.1f}%)  │ {wr_1_1:.0f}% 😰              │")
    print(f"   │ 1:1.5       │ {wr_1_15:.0f}% 😐              │")
    print(f"   │ 1:2         │ {wr_1_2:.0f}% ✅              │")
    print(f"   │ 1:3         │ {wr_1_3:.0f}% 🎯              │")
    
    print(f"   └─────────────┴──────────────────────┘")
    
    # Confirm
    print(f"\n{'━' * 60}")
    confirm = input(f"Execute {side} {qty} {symbol.split('/')[0]} @ ${format_price(info, entry)}? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print(f"{R}❌ Cancelled{RST}")
        return
    
    try:
        # Place order
        print(f"\n{Y}📤 Placing {side} {order_type} order...{RST}")
        
        params = {
            "category": "linear",
            "symbol": linear,
            "side": side,
            "orderType": order_type,
            "qty": qty
        }
        
        if order_type == "Limit":
            params["price"] = format_price(info, limit_price)
            params["timeInForce"] = "GTC"
        
        order = exchange.place_order(**params)
        
        if order.get('retCode') != 0:
            print(f"{R}❌ Order failed: {order.get('retMsg')}{RST}")
            return
        
        order_id = order['result']['orderId']
        print(f"{G}✅ Order placed: {order_id}{RST}")
        
        # Wait for fill and get actual entry price
        actual_entry = entry
        if order_type == "Limit":
            filled, actual_entry = wait_for_limit_fill(exchange, linear, order_id, limit_price, side)
            if not filled:
                return
        else:
            time.sleep(1)
            # For market orders, get actual fill price
            actual_entry = current
        
        # Wait for position to be created
        print(f"{Y}⏳ Waiting for position...{RST}")
        position_created = False
        for i in range(30):  # Wait up to 30 seconds
            pos = exchange.get_positions(category="linear", symbol=linear)
            if pos.get('retCode') == 0 and pos['result']['list']:
                if float(pos['result']['list'][0]['size']) > 0:
                    position_created = True
                    # Get actual entry price from position
                    actual_entry = float(pos['result']['list'][0]['avgPrice'])
                    print(f"{G}✅ Position created at ${format_price(info, actual_entry)}{RST}")
                    break
            time.sleep(1)
        
        if not position_created:
            print(f"{R}❌ Position not created after 30 seconds{RST}")
            return
        
        # Recalculate levels with actual entry price
        if direction == 'long':
            sl = actual_entry * (1 - stop_pct)
            tp = actual_entry * (1 + tp_pct)
        else:
            sl = actual_entry * (1 + stop_pct)
            tp = actual_entry * (1 - tp_pct)
        
        # Set trading stops using Bybit native API
        set_trading_stops(exchange, symbol, linear, info, sl, tp, trail_pct, actual_entry)
        
        # Summary
        print(f"\n{G}{'━' * 60}{RST}")
        print(f"{G}{BOLD}✅ Trade Executed!{RST}")
        print(f"{G}{'━' * 60}{RST}")
        print(f"📊 {side} {qty} {symbol} @ ${format_price(info, actual_entry)}")
        print(f"🎯 Target: {tp_sign}{tp_pct*100:.1f}% @ ${format_price(info, tp)} ({G}+${net_profit:,.2f} net{RST})")
        print(f"🛑 Stop: {sl_sign}{stop_pct*100:.1f}% @ ${format_price(info, sl)} ({R}-${net_loss:,.2f} net{RST})")
        print(f"📈 Trailing: {trail_pct:.2f}% (Bybit native)")
        print(f"{G}{'━' * 60}{RST}")
        
        # Log trade with all details
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'qty': qty,
            'entry_price': entry,
            'actual_entry': actual_entry,
            'stop_loss': sl,
            'take_profit': tp,
            'risk_reward': rr,
            'trailing_pct': trail_pct,  # Store as percentage
            'risk_usdt': risk_usdt,
            'reward_usdt': reward_usdt,
            'net_risk': net_loss,
            'net_reward': net_profit,
            'net_rr': net_rr,
            'balance': balance,
            'position_size_pct': position_size_pct * 100,
            'atr': atr if atr else 0,
            'mode': 'Demo' if Config.DEMO_MODE else 'Live'
        }
        log_trade_to_csv(trade_data)
        
    except Exception as e:
        print(f"{R}❌ Error: {e}{RST}")


def main():
    parser = argparse.ArgumentParser(
        description='🚀 Unified Trading Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('action', choices=['buy', 'sell'], help='Trade direction')
    parser.add_argument('symbol', help='Trading pair (e.g., ETH/USDT)')
    
    # Order type group (mutually exclusive)
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument('--market', action='store_true', help='Market order (default)')
    order_group.add_argument('--limit', type=float, metavar='PRICE', help='Limit order at price')
    
    # Risk parameters
    parser.add_argument('--stop', '--sl', type=float, metavar='%', help='Stop loss %')
    parser.add_argument('--tp', type=float, metavar='%', help='Take profit %')
    parser.add_argument('--rr', type=float, metavar='RATIO', help='Risk/Reward ratio')
    parser.add_argument('--trail', type=float, metavar='%', help='Trailing stop %')
    
    # Mode
    parser.add_argument('--real', action='store_true', help='Real money trading')
    
    args = parser.parse_args()
    
    # Parse
    direction = 'long' if args.action == 'buy' else 'short'
    symbol = args.symbol.upper()
    if '/' not in symbol:
        symbol = f"{symbol}/USDT"
    
    # Order type
    if args.limit:
        order_type = "Limit"
        limit_price = args.limit
    else:
        order_type = "Market"
        limit_price = None
    
    # Real mode check
    if args.real:
        Config.DEMO_MODE = False
        print(f"\n{R}{'='*50}{RST}")
        print(f"{R}{BOLD}⚠️  REAL MONEY TRADING ⚠️{RST}")
        print(f"{R}{'='*50}{RST}")
        if input(f"{BOLD}Type 'REAL' to confirm: {RST}") != 'REAL':
            print(f"{R}❌ Cancelled{RST}")
            sys.exit(0)
    
    # Execute
    place_trade(
        direction=direction,
        symbol=symbol,
        stop_pct=args.stop/100 if args.stop else None,
        rr=args.rr,
        tp_pct=args.tp/100 if args.tp else None,
        trail_pct=args.trail if args.trail else None,  # Keep as percentage
        order_type=order_type,
        limit_price=limit_price
    )


if __name__ == "__main__":
    main()