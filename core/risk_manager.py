import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    """
    ETH/USDT Scalping Risk Manager - Optimized for $3,500 price level
    
    Config Reasoning:
    - 9091 USDT: ~2.59 ETH position at $3,507 - optimal for $8-15 moves
    - 10 USDT threshold: Market fee 0.11% = ~10 USDT, covers fees + profit  
    - 180s max hold: RSI signals on 1-min complete in 1-3 minutes
    - 0.6% emergency stop: Limits max loss to ~55 USDT on 9091 position
    """
    
    def __init__(self):
        self.config = {
            'fixed_position_usdt': 9091,        # Optimized for ETH ~$3,507 level
            'fixed_break_even_threshold': 10,   # Market fee 0.11% coverage  
            'leverage': 10,
            'reward_ratio': 1.5,
            'max_position_time': 180,           # 3min max hold for 1-min signals
            'emergency_stop_pct': 0.006         # 0.6% emergency stop
        }
        self.symbol = os.getenv('TRADING_SYMBOL')
    
    def validate_trade(self, signal, balance, current_price):
        """Validate trade execution - streamlined"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        if stop_distance < 0.0001 or stop_distance > 0.02:
            return False, "Invalid stop distance"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance, entry_price, stop_price):
        """Calculate position size - fixed USDT amount"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        position_usdt = min(self.config['fixed_position_usdt'], balance * 0.5)
        position_size = position_usdt / entry_price
        
        return round(max(position_size, 0), 3)
    
    def should_close_position(self, current_price, entry_price, side, unrealized_pnl, position_age_seconds):
        """Position exit logic - streamlined"""
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        
        # Emergency stop
        if pnl_pct <= -self.config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time  
        if position_age_seconds >= self.config['max_position_time']:
            return True, "max_hold_time_exceeded"
        
        # Fixed take profit threshold
        if unrealized_pnl >= self.config['fixed_break_even_threshold']:
            return True, "profit_lock"
        
        return False, "hold"
    
    def get_leverage(self):
        """Get leverage setting"""
        return self.config['leverage']