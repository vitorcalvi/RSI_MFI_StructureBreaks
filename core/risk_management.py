class RiskManager:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # Position sizing
        self.leverage = 10                 # Leverage multiplier
        self.max_position_size = 0.05      # 5% of balance per trade
        self.risk_per_trade = 0.02         # 2% account risk per trade
        
        # Loss Protection - OPTIMIZED for ZORA volatility
        self.stop_loss_pct = 0.025         # 2.5% price movement = 25% account loss (acceptable for crypto)
        self.loss_switch_threshold = -0.15 # -15% account loss before switching (was -0.5%)
        
        # Profit System and Protection - OPTIMIZED for realistic targets
        self.take_profit_pct = 0.05        # 5% price movement = 50% account gain (2:1 R/R)
        self.break_even_pct = 0.01         # 1% price movement = 10% account gain (profit lock trigger)
        self.trailing_stop_distance = 0.015 # 1.5% price movement = 15% account loss (trailing distance)
        
        # ZORA/USDT specific settings (price around 0.089)
        self.min_price_movement = 0.0001   # Minimum price tick for ZORA
        
        # Advanced risk settings
        self.max_daily_loss = 0.10         # Stop trading if -10% daily loss
        self.max_consecutive_losses = 3    # Stop after 3 consecutive losses
        
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
    
    def get_risk_summary(self, price=0.089):
        """Get comprehensive risk summary for current settings"""
        return {
            'stop_loss_price_long': price * (1 - self.stop_loss_pct),
            'stop_loss_price_short': price * (1 + self.stop_loss_pct),
            'take_profit_price_long': price * (1 + self.take_profit_pct),
            'take_profit_price_short': price * (1 - self.take_profit_pct),
            'break_even_price_long': price * (1 + self.break_even_pct),
            'break_even_price_short': price * (1 - self.break_even_pct),
            'account_risk_per_trade': f"{self.stop_loss_pct * self.leverage * 100:.1f}%",
            'account_reward_potential': f"{self.take_profit_pct * self.leverage * 100:.1f}%",
            'risk_reward_ratio': f"1:{self.take_profit_pct / self.stop_loss_pct:.1f}"
        }
    
    def validate_trade_conditions(self, balance, current_price, position_size_pct):
        """Validate if trade conditions are acceptable"""
        conditions = {
            'sufficient_balance': balance > 100,  # Minimum $100
            'valid_price': current_price > 0.001,  # Reasonable price range
            'acceptable_position_size': 0.01 <= position_size_pct <= 0.10,  # 1-10% position
            'leverage_within_limits': 1 <= self.leverage <= 20  # Reasonable leverage
        }
        
        return all(conditions.values()), conditions
    
    def get_position_metrics(self, entry_price, current_price, side, position_size):
        """Calculate current position metrics"""
        if side.lower() == 'long' or side.lower() == 'buy':
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            price_change_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Account P&L with leverage
        account_pnl_pct = price_change_pct * self.leverage
        
        # Unrealized P&L in USDT
        unrealized_pnl = (position_size * entry_price * price_change_pct / 100)
        
        return {
            'price_change_pct': price_change_pct,
            'account_pnl_pct': account_pnl_pct,
            'unrealized_pnl_usdt': unrealized_pnl,
            'distance_to_stop_loss_pct': abs(price_change_pct - (-self.stop_loss_pct * 100)),
            'distance_to_take_profit_pct': abs(self.take_profit_pct * 100 - price_change_pct)
        }