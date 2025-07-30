import json
import os

class RiskManager:
    def __init__(self):
        self._load_config()
        
        # Set leverage to 25x everywhere
        self.leverage = 25                      # 25x leverage
        self.max_position_size = 0.1          # 0.2% of wallet balance
        self.stop_loss_pct = 0.015              # 1.5% stop loss distance
        self.trailing_stop_distance = 0.01      # 1% trailing distance
        
        # Risk thresholds for 25x leverage
        self.profit_lock_threshold = 12.5       # 12.5% position P&L = 0.5% wallet impact
        self.profit_protection_threshold = 50.0 # 50% position P&L = 2.0% wallet impact
    
    def _load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), 'strategies', 'params_RSI_MFI_Cloud.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    @property
    def symbol(self):
        return self.config['symbol']
    
    def calculate_position_size(self, balance, price):
        """Calculate position size with proper risk management"""
        base_value = balance * self.max_position_size
        return base_value / price
    
    def get_actual_risk_per_trade(self, balance):
        """Calculate actual risk percentage per trade"""
        position_value = balance * self.max_position_size
        max_loss_with_leverage = position_value * self.stop_loss_pct
        return (max_loss_with_leverage / balance) * 100
    
    def should_activate_profit_lock(self, position_pnl_pct):
        """Activate profit lock at 12.5% position P&L"""
        return position_pnl_pct >= self.profit_lock_threshold
    
    def should_take_profit_protection(self, position_pnl_pct):
        """Close at 50% position P&L"""
        return position_pnl_pct >= self.profit_protection_threshold
    
    def get_stop_loss(self, entry_price, side='long'):
        """Calculate stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_trailing_stop_distance_absolute(self, current_price):
        """Calculate absolute trailing stop distance"""
        return current_price * self.trailing_stop_distance
    
    def get_risk_zone(self, position_pnl_pct):
        """Return current risk zone for display"""
        if position_pnl_pct >= self.profit_protection_threshold:
            return "PROFIT_PROTECTION"
        elif position_pnl_pct >= self.profit_lock_threshold:
            return "PROFIT_LOCK"  
        else:
            return "NORMAL"
    
    def get_risk_summary(self, balance):
        """Get risk summary for display"""
        position_value = balance * self.max_position_size
        max_loss = position_value * self.stop_loss_pct
        risk_pct = self.get_actual_risk_per_trade(balance)
        
        return {
            'leverage': self.leverage,
            'position_size_pct': self.max_position_size * 100,
            'position_value': position_value,
            'max_loss_usd': max_loss,
            'risk_per_trade_pct': risk_pct,
            'profit_lock_threshold': self.profit_lock_threshold,
            'profit_protection_threshold': self.profit_protection_threshold,
            'wallet_profit_lock': self.profit_lock_threshold / self.leverage,
            'wallet_profit_protection': self.profit_protection_threshold / self.leverage
        }