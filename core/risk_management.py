class RiskManager:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # Position sizing
        self.leverage = 10
        self.max_position_size = 0.1       # 10% of balance per trade
        self.risk_per_trade = 0.04         # 4% account risk per trade
        
        # Risk levels
        self.stop_loss_pct = 0.035         # 3.5% price movement
        self.loss_switch_threshold = -0.08 # -8% account loss before switching
        
        # Profit levels (optimized for ZORA price ~$0.08)
        self.take_profit_pct = 0.07        # 7% price movement
        self.break_even_pct = 0.01         # 1% price movement (profit lock trigger = 10% account P&L)
        self.trailing_stop_distance = 0.008 # 0.8% trailing distance (tighter for volatile low-price coin)
        
        # Profit protection
        self.profit_protection_threshold = 4.0  # 4% account profit threshold
        
    def calculate_position_size(self, balance, price):
        """Calculate position size based on risk parameters"""
        base_value = balance * self.max_position_size
        
        # Price-based adjustment
        if price < 0.05:
            base_value *= 0.7  # Reduce for very low prices
        elif price > 0.20:
            base_value *= 1.1  # Increase for higher prices
        
        return base_value / price
    
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