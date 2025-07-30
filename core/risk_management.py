class RiskManager:
    def __init__(self):
        # Simple fixed risk approach
        self.symbol = "ETH/USDT"  # Symbol moved here
        self.linear = self.symbol.replace('/', '')
        
        # Fixed risk per trade
        self.fixed_risk_usd = 100.0         # Fixed $100 risk per trade
        self.stop_loss_pct = 0.015          # 1.5% stop loss
        
        # Profit management (simple position P&L)
        self.profit_lock_threshold = 3.0    # 3% position profit to activate trailing
        self.trailing_stop_pct = 0.008      # 0.8% trailing stop
        self.take_profit_pct = 0.06         # 6% take profit (4:1 RR)
    
    def calculate_position_size(self, balance, entry_price):
        """Simple position sizing based on fixed $100 risk"""
        # Calculate position size to risk exactly $100
        stop_distance = entry_price * self.stop_loss_pct
        position_size_tokens = self.fixed_risk_usd / stop_distance
        
        # Position value in USD
        position_value = position_size_tokens * entry_price
        
        # Don't risk more than 10% of balance
        max_position_value = balance * 0.10
        if position_value > max_position_value:
            position_size_tokens = max_position_value / entry_price
            actual_risk = position_size_tokens * stop_distance
        else:
            actual_risk = self.fixed_risk_usd
            
        return position_size_tokens
    
    def get_stop_loss_price(self, entry_price, side='long'):
        """Get stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit_price(self, entry_price, side='long'):
        """Get take profit price"""
        if side == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def should_activate_profit_lock(self, entry_price, current_price, side='long'):
        """Check if should activate profit lock"""
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            
        return profit_pct >= self.profit_lock_threshold
    
    def get_trailing_stop_price(self, current_price, side='long'):
        """Get trailing stop price"""
        if side == 'long':
            return current_price * (1 - self.trailing_stop_pct)
        else:
            return current_price * (1 + self.trailing_stop_pct)
    
    def get_risk_summary(self, balance, entry_price):
        """Get simple risk summary"""
        position_size = self.calculate_position_size(balance, entry_price)
        position_value = position_size * entry_price
        risk_pct = (self.fixed_risk_usd / balance) * 100
        
        return {
            'symbol': self.symbol,
            'fixed_risk_usd': self.fixed_risk_usd,
            'risk_pct': risk_pct,
            'position_size': position_size,
            'position_value': position_value,
            'stop_loss_pct': self.stop_loss_pct * 100,
            'take_profit_pct': self.take_profit_pct * 100,
            'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct
        }