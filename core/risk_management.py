import json
import os

class RiskManager:
    def __init__(self):
        self._load_config()
        
        # SIMPLE FIXED PARAMETERS
        self.leverage = 25
        self.max_position_size = 0.04  # 4% of balance
        self.stop_loss_pct = 0.015     # 1.5%
        self.trailing_stop_distance = 0.008  # 0.8%
        
        # PROFIT THRESHOLDS (% of account)
        self.profit_lock_threshold = 0.8      # 0.8% to activate trailing
        self.profit_protection_threshold = 2.0 # 2.0% to close
        self.profit_reversal_threshold = 1.0   # 1.0% profit to reverse
        self.loss_reversal_threshold = -1.0    # -1.0% loss to reverse
        
        # COOLDOWN
        self.reversal_cooldown_cycles = self.config.get('signal_cooldown', 2)
    
    def _load_config(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), 'strategies', 'params_RSI_MFI_Cloud.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                'oversold_level': 40,
                'overbought_level': 50,
                'signal_cooldown': 1
            }
    
    @property
    def symbol(self):
        return self.config.get('symbol', 'ZORA/USDT')
    
    def calculate_account_pnl_pct(self, unrealized_pnl, account_balance):
        return (unrealized_pnl / account_balance) * 100 if account_balance > 0 else 0
    
    def calculate_position_size(self, balance, price):
        base_value = balance * self.max_position_size
        return base_value / price
    
    def should_activate_profit_lock(self, account_pnl_pct):
        return account_pnl_pct >= self.profit_lock_threshold
    
    def should_take_profit_protection(self, account_pnl_pct):
        return account_pnl_pct >= self.profit_protection_threshold
    
    def should_reverse_for_profit(self, account_pnl_pct):
        return account_pnl_pct >= self.profit_reversal_threshold
    
    def should_reverse_for_loss(self, account_pnl_pct):
        return account_pnl_pct <= self.loss_reversal_threshold
    
    def get_stop_loss(self, entry_price, side='long'):
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_trailing_stop_distance_absolute(self, current_price):
        return current_price * self.trailing_stop_distance