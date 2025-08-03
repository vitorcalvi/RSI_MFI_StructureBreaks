import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    """ETH/USDT Scalping Risk Manager - $3,500 optimized"""

    def __init__(self):
        self.config = {
            'fixed_position_usdt': 9091,
            'fixed_break_even_threshold': 15,
            'leverage': 10,
            'reward_ratio': 1.5,
            'max_position_time': 180,
            'emergency_stop_amount': 5000
        }
        self.symbol = os.getenv('TRADING_SYMBOL')
    
    def validate_trade(self, signal, balance, current_price):
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        if stop_distance < 0.0001 or stop_distance > 0.02:
            return False, "Invalid stop distance"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance, entry_price, stop_price):
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        position_usdt = min(self.config['fixed_position_usdt'], balance * 0.5)
        return round(position_usdt / entry_price, 3)
    
    def should_close_position(self, current_price, entry_price, side, unrealized_pnl, position_age_seconds):
        if unrealized_pnl <= -self.config['emergency_stop_amount']:
            return True, "emergency_stop"
        
        if position_age_seconds >= self.config['max_position_time']:
            return True, "max_hold_time_exceeded"
        
        if unrealized_pnl >= self.config['fixed_break_even_threshold']:
            return True, "profit_lock"
        
        return False, "hold"
    
    def get_leverage(self):
        return self.config['leverage']