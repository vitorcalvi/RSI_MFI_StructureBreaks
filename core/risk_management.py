class RiskManager:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # Position sizing
        self.leverage = 10                 # Leverage multiplier
        self.max_position_size = 0.05      # 5% of balance per trade
        self.risk_per_trade = 0.02         # 2% account risk per trade
        
        # Loss Protection  
        self.stop_loss_pct = 0.008         # 0.8% price movement = 8% account loss
        self.loss_switch_threshold = -0.005 # -0.5% price movement = 5% account loss
        
        # Profit System and Protection
        self.take_profit_pct = 0.016       # 1.6% price movement = 16% account gain  
        self.break_even_pct = 0.002        # 0.2% price movement = 2% account gain
        self.trailing_stop_distance = 0.003 # 0.3% price movement = 3% account loss
        
        # ZORA/USDT specific settings (price around 0.082)
        self.min_price_movement = 0.0001   # Minimum price tick for ZORA
        
    def calculate_position_size(self, balance, price):
        """Calculate position size based on risk parameters"""
        max_value = balance * self.max_position_size
        position_size = max_value / price
        return position_size
    
    def calculate_risk_amount(self, balance):
        """Calculate maximum risk amount per trade"""
        return balance * self.risk_per_trade
    
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
    
    def get_trailing_stop_distance_pct(self):
        """Get trailing stop as percentage (for display)"""
        return self.trailing_stop_distance * 100  # Convert to percentage