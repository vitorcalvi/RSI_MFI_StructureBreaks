class RiskManager:
    def __init__(self):
        self.max_position_size = 0.05  # 5% of balance (reduced from 10%)
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.04    # 4% take profit
        
    def calculate_position_size(self, balance, price):
        """Calculate position size based on risk parameters"""
        max_value = balance * self.max_position_size
        position_size = max_value / price
        return position_size
    
    def get_stop_loss(self, entry_price, side='long'):
        """Calculate stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit(self, entry_price, side='long'):
        """Calculate take profit price"""
        if side == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)