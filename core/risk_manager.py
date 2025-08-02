import os
from dotenv import load_dotenv

load_dotenv()


"""
1-Minute Trading Config (FIXED)

Values Reasoning:
- 7500 USDT: Large enough to profit from small 1-min moves (~2.14 ETH position)
- 12 USDT threshold: Trading fees = 1.8 USDT, need 12+ for true profit
- 180 seconds: RSI signals on 1-min complete in 1-3 minutes typically
- 1.5 reward ratio: Realistic for quick scalping vs 2.5+ for swing trades
- 0.6% stop: Limits max loss to ~45 USDT, prevents blown account
- 55% win rate needed for break-even with these settings
"""

class RiskManager:
    def __init__(self):
        self.config = {
    'fixed_position_usdt': 9091,       # Fixed position size - optimized for 1-min $5-8 moves
    # Round Market fee 0.11% 
    'fixed_break_even_threshold': 10,  # Min profit to close - covers fees + small profit
    'leverage': 10,                    # Standard leverage
    'reward_ratio': 1.5,              # Risk/reward for 1-min scalping
    'max_position_time': 180,          # 3min max hold for 1-min signals
    'emergency_stop_pct': 0.006        # 0.6% emergency stop
}

        self.symbol = os.getenv('TRADING_SYMBOL')
    
    def validate_trade(self, signal, balance, current_price):
        """Validate if trade should be executed"""
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
        
        position_usdt = min(self.config['fixed_position_usdt'], balance * 0.5)
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
        
        # Fixed take profit threshold ($11)
        if unrealized_pnl >= self.config['fixed_break_even_threshold']:
            return True, "profit_lock"
        
        return False, "hold"
    
    def get_leverage(self):
        """Get leverage setting"""
        return self.config['leverage']