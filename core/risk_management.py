class RiskManager:
    def __init__(self):
        # Simple fixed risk approach
        self.symbol = "ETH/USDT"
        self.linear = self.symbol.replace('/', '')
        
        # Fixed risk per trade
        self.fixed_risk_usd = 100.0         # Fixed $100 risk per trade
        self.stop_loss_pct = 0.015          # 1.5% stop loss (fallback only)
        
        # Profit management
        self.profit_lock_threshold = 3.0    # 3% position profit to activate trailing
        self.trailing_stop_pct = 0.008      # 0.8% trailing stop
        self.take_profit_pct = 0.06         # 6% take profit (4:1 RR)
        
        # Break & Retest enhancements
        self.retest_risk_multiplier = 1.5   # 1.5x risk for high-probability retests
        self.retest_reward_multiplier = 1.5 # 1.5x reward targets for retests
        self.max_retest_risk_usd = 200.0    # Cap retest risk at $200
        
        print("✅ Structure-Based Stop Loss")
        print("🎯 Break & Retest risk enhancement enabled")
    
    def calculate_position_size(self, balance, entry_price, structure_stop=None, signal_type=None):
        """Enhanced position size calculation with Break & Retest support"""
        if structure_stop:
            # Use actual structure stop distance
            stop_distance = abs(entry_price - structure_stop)
        else:
            # Fallback to fixed percentage
            stop_distance = entry_price * self.stop_loss_pct
        
        # Base risk amount
        base_risk = self.fixed_risk_usd
        
        # Enhanced risk for Break & Retest patterns
        if signal_type == 'BREAK_RETEST':
            enhanced_risk = min(base_risk * self.retest_risk_multiplier, self.max_retest_risk_usd)
            print(f"🎯 Enhanced Risk | Base: ${base_risk:.0f} → Retest: ${enhanced_risk:.0f}")
            risk_amount = enhanced_risk
        else:
            risk_amount = base_risk
        
        # Calculate position size to risk the determined amount
        position_size_tokens = risk_amount / stop_distance
        
        # Position value in USD
        position_value = position_size_tokens * entry_price
        
        # Don't risk more than 10% of balance (15% for high-confidence retests)
        max_risk_pct = 0.15 if signal_type == 'BREAK_RETEST' else 0.10
        max_position_value = balance * max_risk_pct
        
        if position_value > max_position_value:
            position_size_tokens = max_position_value / entry_price
            actual_risk = position_size_tokens * stop_distance
            print(f"⚠️ Position Capped | Max {max_risk_pct*100:.0f}% of balance | Risk reduced to ${actual_risk:.0f}")
        else:
            actual_risk = risk_amount
            
        return position_size_tokens
    
    def get_stop_loss_price(self, entry_price, side='long', structure_stop=None):
        """Get stop loss price - structure-based if available"""
        if structure_stop:
            return structure_stop
        
        # Fallback to fixed percentage
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit_price(self, entry_price, side='long', structure_stop=None, signal_type=None):
        """Enhanced take profit calculation with Break & Retest targets"""
        if structure_stop:
            # Calculate actual risk distance
            risk_distance = abs(entry_price - structure_stop)
            
            # Enhanced reward for Break & Retest patterns
            if signal_type == 'BREAK_RETEST':
                # 6:1 reward ratio for high-confidence retests
                reward_distance = risk_distance * 6
                print(f"🎯 Enhanced Target | 6:1 R:R for Break & Retest pattern")
            else:
                # Standard 4:1 reward ratio
                reward_distance = risk_distance * 4
            
            if side == 'long':
                return entry_price + reward_distance
            else:
                return entry_price - reward_distance
        
        # Fallback to fixed percentage with enhancement
        base_tp_pct = self.take_profit_pct
        
        if signal_type == 'BREAK_RETEST':
            # 1.5x normal target for retests
            enhanced_tp_pct = base_tp_pct * self.retest_reward_multiplier
            print(f"🎯 Enhanced Target | {enhanced_tp_pct*100:.1f}% for Break & Retest")
            
            if side == 'long':
                return entry_price * (1 + enhanced_tp_pct)
            else:
                return entry_price * (1 - enhanced_tp_pct)
        
        # Standard targets
        if side == 'long':
            return entry_price * (1 + base_tp_pct)
        else:
            return entry_price * (1 - base_tp_pct)
    
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
    
    def get_risk_summary(self, balance, entry_price, structure_stop=None, signal_type=None):
        """Enhanced risk summary with Break & Retest info"""
        position_size = self.calculate_position_size(balance, entry_price, structure_stop, signal_type)
        position_value = position_size * entry_price
        
        # Calculate actual risk
        if structure_stop:
            actual_risk = position_size * abs(entry_price - structure_stop)
            stop_distance_pct = (abs(entry_price - structure_stop) / entry_price) * 100
        else:
            if signal_type == 'BREAK_RETEST':
                actual_risk = min(self.fixed_risk_usd * self.retest_risk_multiplier, self.max_retest_risk_usd)
            else:
                actual_risk = self.fixed_risk_usd
            stop_distance_pct = self.stop_loss_pct * 100
        
        risk_pct = (actual_risk / balance) * 100
        
        # Enhanced info for Break & Retest
        enhancement_info = {}
        if signal_type == 'BREAK_RETEST':
            enhancement_info = {
                'enhanced_pattern': True,
                'risk_multiplier': self.retest_risk_multiplier,
                'reward_multiplier': 1.5,  # 6:1 vs 4:1
                'confidence_level': 'HIGH'
            }
        
        return {
            'symbol': self.symbol,
            'fixed_risk_usd': self.fixed_risk_usd,
            'actual_risk_usd': actual_risk,
            'risk_pct': risk_pct,
            'position_size': position_size,
            'position_value': position_value,
            'stop_distance_pct': stop_distance_pct,
            'structure_based': structure_stop is not None,
            'structure_stop': structure_stop,
            'signal_type': signal_type,
            'enhancement_info': enhancement_info
        }