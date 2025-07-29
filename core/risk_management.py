import json
import os

class RiskManager:
    def __init__(self):
        self._load_config()
        
        self.leverage = 25                      # FIXED: 25x to trigger conditions to test 
        self.max_position_size = 0.002          # FIXED: 0.2% for 5% total risk with 25x leverage
        self.stop_loss_pct = 0.015              # 1.5% stop loss distance
        self.trailing_stop_distance = 0.01      # 1% trailing distance
        
        # Risk thresholds (based on wallet balance P&L)
        self.profit_lock_threshold = 0.5        # 0.5% wallet P&L to activate trailing
        self.profit_protection_threshold = 2.0  # 2.0% wallet P&L to close
        self.loss_reversal_threshold = -1.0     # -1.0% wallet P&L to reverse
        
        # Cooldown cycles
        self.reversal_cooldown_cycles = self.config.get('signal_cooldown', 2)
    
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
        max_loss = position_value * self.stop_loss_pct
        return (max_loss / balance) * 100
    
    def should_activate_profit_lock(self, wallet_pnl_pct):
        """Activate profit lock at 0.5% wallet P&L"""
        return wallet_pnl_pct >= self.profit_lock_threshold
    
    def should_take_profit_protection(self, wallet_pnl_pct):
        """HIGHEST PRIORITY - Close at 2.0% wallet P&L"""
        return wallet_pnl_pct >= self.profit_protection_threshold
    
    def should_reverse_for_loss(self, wallet_pnl_pct):
        """Only reverse on loss - NOT profit"""
        return wallet_pnl_pct <= self.loss_reversal_threshold
    
    def get_stop_loss(self, entry_price, side='long'):
        """Calculate stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_trailing_stop_distance_absolute(self, current_price):
        """Calculate absolute trailing stop distance"""
        return current_price * self.trailing_stop_distance
    
    def get_risk_zone(self, wallet_pnl_pct):
        """Return current risk zone for display"""
        if wallet_pnl_pct >= self.profit_protection_threshold:
            return "PROFIT_PROTECTION"
        elif wallet_pnl_pct >= self.profit_lock_threshold:
            return "PROFIT_LOCK"
        elif wallet_pnl_pct <= self.loss_reversal_threshold:
            return "LOSS_REVERSAL"
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
            'loss_reversal_threshold': self.loss_reversal_threshold
        }