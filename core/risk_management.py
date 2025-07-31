import json
import os

class RiskManager:
    def __init__(self):
        self.symbol = "ETH/USDT"
        self.linear = self.symbol.replace('/', '')
        self._load_params()
        
        # Maintain backward compatibility with existing code
        self.fixed_risk_usd = 100.0  # Keep original fixed USD risk
        self.stop_loss_pct = 0.015   # 1.5% fallback
        self.take_profit_pct = 0.06  # 6% fallback
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.01)
        self.profit_lock_threshold = self.params.get('profit_lock_threshold', 4.0)
        
        # Break & Retest enhancements
        self.retest_risk_multiplier = 1.5
        self.retest_reward_multiplier = 1.5
        self.max_retest_risk_usd = 200.0
        
        print("✅ Structure-Based Stop Loss")
        print("🎯 Break & Retest risk enhancement enabled")
    
    def _load_params(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            params_file = os.path.join(current_dir, '..', 'strategies', 'params_RSI_MFI_Cloud.json')
            with open(params_file, 'r') as f:
                self.params = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load params file: {e}")
            # Fallback parameters
            self.params = {
                'fixed_risk_pct': 0.02,
                'trailing_stop_pct': 0.01,
                'profit_lock_threshold': 4.0,
                'reward_ratio': 5.0,
                'entry_fee_pct': 0.00055,
                'exit_fee_pct': 0.00055
            }
    
    def calculate_position_size(self, balance, entry_price, structure_stop=None, signal_type=None):
        """Enhanced position size calculation with dual approach"""
        if structure_stop:
            stop_distance = abs(entry_price - structure_stop)
        else:
            stop_distance = entry_price * self.stop_loss_pct
        
        # Use percentage-based risk from new params if available
        if 'fixed_risk_pct' in self.params:
            risk_amount = balance * self.params['fixed_risk_pct']
        else:
            # Fallback to fixed USD risk
            risk_amount = self.fixed_risk_usd
        
        # Enhanced risk for Break & Retest patterns
        if signal_type == 'BREAK_RETEST':
            if 'fixed_risk_pct' in self.params:
                enhanced_risk = min(risk_amount * self.retest_risk_multiplier, balance * 0.05)  # Cap at 5%
            else:
                enhanced_risk = min(risk_amount * self.retest_risk_multiplier, self.max_retest_risk_usd)
            print(f"🎯 Enhanced Risk | Base: ${risk_amount:.0f} → Retest: ${enhanced_risk:.0f}")
            risk_amount = enhanced_risk
        
        # Calculate position size
        position_size_tokens = risk_amount / stop_distance
        position_value = position_size_tokens * entry_price
        
        # Position caps
        max_risk_pct = 0.15 if signal_type == 'BREAK_RETEST' else 0.10
        max_position_value = balance * max_risk_pct
        
        if position_value > max_position_value:
            position_size_tokens = max_position_value / entry_price
            actual_risk = position_size_tokens * stop_distance
            print(f"⚠️ Position Capped | Max {max_risk_pct*100:.0f}% of balance | Risk: ${actual_risk:.0f}")
        
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
        """Enhanced take profit calculation"""
        if structure_stop:
            risk_distance = abs(entry_price - structure_stop)
            
            # Get reward ratio from params or use default
            reward_ratio = self.params.get('reward_ratio', 4.0)
            
            # Enhanced reward for Break & Retest patterns
            if signal_type == 'BREAK_RETEST':
                reward_ratio *= self.retest_reward_multiplier  # 1.5x multiplier
                print(f"🎯 Enhanced Target | {reward_ratio:.1f}:1 R:R for Break & Retest")
            
            reward_distance = risk_distance * reward_ratio
            
            if side == 'long':
                return entry_price + reward_distance
            else:
                return entry_price - reward_distance
        
        # Fallback to percentage-based
        base_tp_pct = self.take_profit_pct
        
        if signal_type == 'BREAK_RETEST':
            base_tp_pct *= self.retest_reward_multiplier
            print(f"🎯 Enhanced Target | {base_tp_pct*100:.1f}% for Break & Retest")
        
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
        
        threshold = self.params.get('profit_lock_threshold', self.profit_lock_threshold)
        return profit_pct >= threshold
    
    def get_trailing_stop_price(self, current_price, side='long'):
        """Get trailing stop price"""
        trail_pct = self.params.get('trailing_stop_pct', self.trailing_stop_pct)
        if side == 'long':
            return current_price * (1 - trail_pct)
        else:
            return current_price * (1 + trail_pct)
    
    def get_risk_summary(self, balance, entry_price, structure_stop=None, signal_type=None):
        """Enhanced risk summary with parameter info"""
        position_size = self.calculate_position_size(balance, entry_price, structure_stop, signal_type)
        position_value = position_size * entry_price
        
        # Calculate actual risk
        if structure_stop:
            actual_risk = position_size * abs(entry_price - structure_stop)
            stop_distance_pct = (abs(entry_price - structure_stop) / entry_price) * 100
        else:
            if 'fixed_risk_pct' in self.params:
                base_risk = balance * self.params['fixed_risk_pct']
            else:
                base_risk = self.fixed_risk_usd
            
            if signal_type == 'BREAK_RETEST':
                actual_risk = min(base_risk * self.retest_risk_multiplier, 
                                balance * 0.05 if 'fixed_risk_pct' in self.params else self.max_retest_risk_usd)
            else:
                actual_risk = base_risk
            stop_distance_pct = self.stop_loss_pct * 100
        
        risk_pct = (actual_risk / balance) * 100
        
        # Enhancement info for Break & Retest
        enhancement_info = {}
        if signal_type == 'BREAK_RETEST':
            enhancement_info = {
                'enhanced_pattern': True,
                'risk_multiplier': self.retest_risk_multiplier,
                'reward_multiplier': self.retest_reward_multiplier,
                'confidence_level': 'HIGH'
            }
        
        return {
            'symbol': self.symbol,
            'risk_approach': 'percentage' if 'fixed_risk_pct' in self.params else 'fixed_usd',
            'base_risk': self.params.get('fixed_risk_pct', self.fixed_risk_usd),
            'actual_risk_usd': actual_risk,
            'risk_pct': risk_pct,
            'position_size': position_size,
            'position_value': position_value,
            'stop_distance_pct': stop_distance_pct,
            'structure_based': structure_stop is not None,
            'structure_stop': structure_stop,
            'signal_type': signal_type,
            'enhancement_info': enhancement_info,
            'reward_ratio': self.params.get('reward_ratio', 4.0),
            'profit_lock_threshold': self.params.get('profit_lock_threshold', self.profit_lock_threshold),
            'trailing_stop_pct': self.params.get('trailing_stop_pct', self.trailing_stop_pct)
        }