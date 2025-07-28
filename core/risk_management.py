class RiskManager:
    def __init__(self):
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # Position sizing - OPTIMIZED for ZORA volatility
        self.leverage = 10                 # Leverage multiplier
        self.max_position_size = 0.05      # 5% of balance per trade (matches your current setting)
        self.risk_per_trade = 0.02         # 2% account risk per trade (matches your current setting)
        
        # Loss Protection - OPTIMIZED for ZORA's volatility
        self.stop_loss_pct = 0.035         # 3.5% price movement = 35% account loss with 10x leverage
        self.loss_switch_threshold = -0.08 # -8% account loss before switching (matches your log)
        
        # Profit System - OPTIMIZED for ZORA's explosive moves
        self.take_profit_pct = 0.07        # 7% price movement = 70% account gain (2:1 R/R)
        self.break_even_pct = 0.02         # 2% price movement = 20% account gain (profit lock trigger)
        self.trailing_stop_distance = 0.02 # 2% price movement (matches your settings)
        
        # ZORA/USDT specific settings
        self.min_price_movement = 0.0001   # Minimum price tick for ZORA
        
        # Advanced risk settings
        self.max_daily_loss = 0.15         # Stop trading if -15% daily loss
        self.max_consecutive_losses = 3    # Stop after 3 consecutive losses
        
        # Profit protection threshold
        self.profit_protection_threshold = 4.0  # 4% account profit - don't reverse
        
        # ZORA-specific volatility adjustments
        self.volatility_multiplier = 1.2   # 20% wider stops during high volatility periods
        self.quick_profit_target = 0.03    # 3% quick profit target for scalping
        
    def calculate_position_size(self, balance, price):
        """Calculate position size based on risk parameters - ZORA optimized"""
        # Base position size
        base_value = balance * self.max_position_size
        
        # ZORA-specific adjustment: Reduce size if price < $0.05 (higher risk)
        if price < 0.05:
            base_value *= 0.7  # 30% reduction for very low prices
        elif price > 0.20:
            base_value *= 1.1  # 10% increase for higher prices (more stable)
        
        position_size = base_value / price
        return position_size
    
    def calculate_risk_amount(self, balance):
        """Calculate maximum risk amount per trade"""
        return balance * self.risk_per_trade
    
    def get_stop_loss(self, entry_price, side='long'):
        """Calculate stop loss price - ZORA volatility adjusted"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit(self, entry_price, side='long'):
        """Calculate take profit price - ZORA explosive move optimized"""
        if side == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def get_quick_profit_target(self, entry_price, side='long'):
        """ZORA-specific: Quick profit target for scalping volatile moves"""
        if side == 'long':
            return entry_price * (1 + self.quick_profit_target)
        else:
            return entry_price * (1 - self.quick_profit_target)
    
    def get_trailing_stop_distance_pct(self):
        """Get trailing stop as percentage (for display)"""
        return self.trailing_stop_distance * 100  # Convert to percentage
    
    def get_risk_summary(self, price=0.086):
        """Get comprehensive risk summary for ZORA trading"""
        return {
            'stop_loss_price_long': price * (1 - self.stop_loss_pct),
            'stop_loss_price_short': price * (1 + self.stop_loss_pct),
            'take_profit_price_long': price * (1 + self.take_profit_pct),
            'take_profit_price_short': price * (1 - self.take_profit_pct),
            'quick_profit_price_long': price * (1 + self.quick_profit_target),
            'quick_profit_price_short': price * (1 - self.quick_profit_target),
            'break_even_price_long': price * (1 + self.break_even_pct),
            'break_even_price_short': price * (1 - self.break_even_pct),
            'account_risk_per_trade': f"{self.stop_loss_pct * self.leverage * 100:.1f}%",
            'account_reward_potential': f"{self.take_profit_pct * self.leverage * 100:.1f}%",
            'risk_reward_ratio': f"1:{self.take_profit_pct / self.stop_loss_pct:.1f}"
        }
    
    def validate_trade_conditions(self, balance, current_price, position_size_pct):
        """Validate if trade conditions are acceptable - ZORA specific"""
        conditions = {
            'sufficient_balance': balance > 100,  # Minimum $100
            'valid_price': 0.001 < current_price < 1.0,  # ZORA price range
            'acceptable_position_size': 0.005 <= position_size_pct <= 0.1,  # 0.5-10% position
            'leverage_within_limits': 5 <= self.leverage <= 20,  # ZORA-appropriate leverage
            'not_too_volatile': True  # Could add volatility check here
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
            'distance_to_take_profit_pct': abs(self.take_profit_pct * 100 - price_change_pct),
            'quick_profit_distance_pct': abs(self.quick_profit_target * 100 - price_change_pct)
        }