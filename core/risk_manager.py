import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    def __init__(self):
        self.config = {
            'fixed_position_usdt': 10000.0,   
            'reward_ratio': 2.5,
            'max_position_time': 75,
            'emergency_stop_pct': 0.006,
            'profit_lock_threshold': 0.002,
            'trailing_stop_pct': 0.0015,
            'entry_fee_pct': 0.00055,
            'exit_fee_pct': 0.00055,
            'min_balance': 10
        }
        
        self.symbol = os.getenv('TRADING_SYMBOL')
    
    def validate_trade(self, signal, balance, current_price):
        """Validate if trade should be executed"""
        if balance < self.config['min_balance']:
            return False, "Insufficient balance"
        
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        if stop_distance < 0.0001 or stop_distance > 0.02:
            return False, "Invalid stop distance"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance, entry_price, stop_price):
        """Calculate position size based on fixed USDT amount"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Use fixed position size instead of percentage-based risk
        position_usdt = min(self.config['fixed_position_usdt'], balance * 0.5)  # Cap at 50% of balance
        position_size = position_usdt / entry_price
        
        return round(max(position_size, 0), 3)
    
    def should_close_position(self, current_price, entry_price, side, unrealized_pnl, position_age_seconds):
        """Determine if position should be closed"""
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        
        # Emergency stop
        if pnl_pct <= -self.config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= self.config['max_position_time']:
            return True, "max_hold_time_exceeded"
        
        # Profit lock
        if pnl_pct >= self.config['profit_lock_threshold']:
            return True, "profit_lock"
        
        # Trailing stop
        price_change = (current_price - entry_price) / entry_price if side == "Buy" else (entry_price - current_price) / entry_price
        if price_change < -self.config['trailing_stop_pct']:
            return True, "trailing_stop"
        
        return False, "hold"