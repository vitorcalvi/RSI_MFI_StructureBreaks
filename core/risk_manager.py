import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    def __init__(self):
        # Hard-coded risk management parameters
        self.config = {
            'fixed_risk_pct': 0.005,
            'reward_ratio': 1.5,
            'max_position_time': 121,
            'emergency_stop_pct': 0.02,
            'profit_lock_threshold': 0.003,
            'trailing_stop_pct': 0.005,
            'entry_fee_pct': 0.00055,
            'exit_fee_pct': 0.00055,
            'min_balance': 10
        }
        
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
    
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
        """Calculate position size based on risk"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        risk_amount = balance * self.config['fixed_risk_pct']
        price_diff = abs(entry_price - stop_price)
        
        if price_diff <= 0:
            return 0
        
        total_fees = self.config['entry_fee_pct'] + self.config['exit_fee_pct']
        position_size = (risk_amount / price_diff) / (1 + total_fees)
        position_size = round(position_size, 3)
        
        # Cap at 10% of balance
        max_size = balance * 0.1 / entry_price
        return min(max(position_size, 0), max_size)
    
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