class RiskManager:
    def __init__(self):
        # =============================================
        # EXPLICIT RISK MANAGEMENT PARAMETERS
        # =============================================
        
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # Leverage & Position Sizing
        self.leverage = 10                    # 10x leverage
        self.max_position_size = 0.1          # 10% of balance per trade
        self.risk_per_trade = 0.04            # 4% account risk per trade
        
        # Price Movement Thresholds (as percentages)
        self.stop_loss_pct = 0.035            # 3.5% price movement = stop loss
        self.take_profit_pct = 0.07           # 7% price movement = take profit
        self.break_even_pct = 0.01            # 1% price movement = profit lock trigger
        self.trailing_stop_distance = 0.008  # 0.8% trailing distance
        
        # Account P&L Thresholds (as percentages of total account)
        self.profit_lock_threshold = 1.0      # 1% account P&L = activate trailing stop
        self.profit_protection_threshold = 4.0 # 4% account P&L = take profit & cooldown
        self.loss_switch_threshold = -8.0     # -8% account P&L = reverse position
        self.position_reversal_threshold = -5.0 # -5% account P&L = reverse on signal
        
        # Cooldown & Control
        self.reversal_cooldown_cycles = 3     # 3 cycles cooldown after profit protection
        
        # Price-based Position Adjustments
        self.low_price_threshold = 0.05       # Below $0.05 = reduce position 30%
        self.high_price_threshold = 0.20      # Above $0.20 = increase position 10%
        self.low_price_multiplier = 0.7       # 70% of normal size for low prices
        self.high_price_multiplier = 1.1      # 110% of normal size for high prices
    
    def calculate_account_pnl_pct(self, unrealized_pnl, account_balance):
        """Calculate P&L as percentage of total account balance"""
        if account_balance <= 0:
            return 0.0
        return (unrealized_pnl / account_balance) * 100
    
    def calculate_position_size(self, balance, price):
        """Calculate position size based on risk parameters"""
        base_value = balance * self.max_position_size
        
        # Price-based adjustment
        if price < self.low_price_threshold:
            base_value *= self.low_price_multiplier
        elif price > self.high_price_threshold:
            base_value *= self.high_price_multiplier
        
        return base_value / price
    
    def should_activate_profit_lock(self, account_pnl_pct):
        """Check if profit lock should be activated"""
        return account_pnl_pct >= self.profit_lock_threshold
    
    def should_take_profit_protection(self, account_pnl_pct):
        """Check if profit protection should trigger"""
        return account_pnl_pct >= self.profit_protection_threshold
    
    def should_switch_position(self, account_pnl_pct):
        """Check if position should be switched due to loss"""
        return account_pnl_pct <= self.loss_switch_threshold
    
    def should_reverse_on_signal(self, account_pnl_pct):
        """Check if position should reverse on opposite signal"""
        return account_pnl_pct <= self.position_reversal_threshold
    
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
    
    def get_trailing_stop_distance_absolute(self, current_price):
        """Calculate absolute trailing stop distance"""
        return current_price * self.trailing_stop_distance
    
    def get_risk_summary(self, balance, current_price):
        """Get comprehensive risk summary for display"""
        position_size = self.calculate_position_size(balance, current_price)
        notional_value = position_size * current_price
        margin_used = notional_value / self.leverage
        
        # Calculate levels
        sl_long = self.get_stop_loss(current_price, 'long')
        sl_short = self.get_stop_loss(current_price, 'short')
        tp_long = self.get_take_profit(current_price, 'long')
        tp_short = self.get_take_profit(current_price, 'short')
        
        return {
            'balance': balance,
            'position_size': position_size,
            'notional_value': notional_value,
            'margin_used': margin_used,
            'margin_pct': (margin_used / balance) * 100,
            'leverage': self.leverage,
            'stop_loss_long': sl_long,
            'stop_loss_short': sl_short,
            'take_profit_long': tp_long,
            'take_profit_short': tp_short,
            'profit_lock_threshold': self.profit_lock_threshold,
            'profit_protection_threshold': self.profit_protection_threshold,
            'loss_switch_threshold': self.loss_switch_threshold,
            'trailing_distance_pct': self.trailing_stop_distance * 100,
            'trailing_distance_abs': self.get_trailing_stop_distance_absolute(current_price)
        }