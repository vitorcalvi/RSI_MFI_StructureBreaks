import json
import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    def __init__(self, config_file="strategies/rsi_mfi.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
        
        print("⚡ Risk Manager initialized")
        print(f"🎯 Symbol: {self.symbol}")
        print(f"💰 Risk per trade: {self.config['fixed_risk_pct']*100}%")
        print(f"🎯 Reward ratio: {self.config['reward_ratio']}:1")
        print(f"⏱️ Max hold time: {self.config['max_position_time']}s")
    
    def _load_config(self):
        """Load strategy configuration with fallback"""
        default_config = {
            "fixed_risk_pct": 0.005, "reward_ratio": 1.5, "max_position_time": 121,
            "emergency_stop_pct": 0.02, "profit_lock_threshold": 0.003, "trailing_stop_pct": 0.005,
            "entry_fee_pct": 0.00055, "exit_fee_pct": 0.00055, "min_balance": 10
        }
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config load error: {e}")
            return default_config
    
    def validate_trade(self, signal, balance, current_price):
        """Validate if trade should be executed"""
        if balance < self.config.get('min_balance', 10):
            return False, "Insufficient balance"
        
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        if stop_distance < 0.0001:
            return False, "Stop loss too tight"
        if stop_distance > 0.02:
            return False, "Stop loss too wide"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance, entry_price, stop_price):
        """Calculate position size based on risk management rules"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        risk_amount = balance * self.config['fixed_risk_pct']
        price_diff = abs(entry_price - stop_price)
        
        if price_diff <= 0:
            return 0
        
        total_fees = self.config['entry_fee_pct'] + self.config['exit_fee_pct']
        position_size = (risk_amount / price_diff) / (1 + total_fees)
        position_size = round(position_size, 3)
        
        if position_size < 0.001:
            return 0
        
        # Cap at 10% of balance
        max_size = balance * 0.1 / entry_price
        return min(position_size, max_size)
    
    def should_close_position(self, current_price, entry_price, side, unrealized_pnl, position_age_seconds):
        """Determine if position should be closed based on risk rules"""
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        
        # Emergency stop
        if pnl_pct <= -self.config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= self.config['max_position_time']:
            return True, "max_hold_time_exceeded"
        
        # Profit lock
        if pnl_pct >= self.config.get('profit_lock_threshold', 0.003):
            return True, "profit_lock"
        
        # Trailing stop
        trailing_threshold = self.config.get('trailing_stop_pct', 0.005)
        price_change = (current_price - entry_price) / entry_price if side == "Buy" else (entry_price - current_price) / entry_price
        
        if price_change < -trailing_threshold:
            return True, "trailing_stop"
        
        return False, "hold"
    
    def get_take_profit_price(self, entry_price, stop_price, side):
        """Calculate take profit price based on reward ratio"""
        risk_distance = abs(entry_price - stop_price)
        reward_distance = risk_distance * self.config['reward_ratio']
        
        return entry_price + reward_distance if side == "Buy" else entry_price - reward_distance
    
    def update_config(self, new_config):
        """Update configuration"""
        try:
            self.config.update(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("✅ Risk config updated")
        except Exception as e:
            print(f"❌ Config update error: {e}")