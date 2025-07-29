# core/risk_management.py - SAFE TRADING PARAMETERS

import json
import os

class RiskManager:
    def __init__(self):
        # Load strategy parameters (centralized config)
        self._load_config()
        
        # =============================================
        # SAFE TRADING PARAMETERS - FIXED
        # =============================================
        
        # SAFE Leverage & Position Sizing
        self.leverage = 5                    # REDUCED from 10x to 5x
        self.max_position_size = 0.04        # REDUCED from 10% to 4%
        self.risk_per_trade = 0.02           # REDUCED from 5% to 2%
        
        # SAFE Price Movement Thresholds
        self.stop_loss_pct = 0.015           # 1.5% price SL (tighter)
        # REMOVED: take_profit_pct - Project holds until signal/protection (no TP)
        self.trailing_stop_distance = 0.008  # 0.8% trailing
        
        # SAFE Account P&L Thresholds - FIXED CONFLICTS
        self.profit_lock_threshold = 0.8     # 0.8% account (realistic with 5x leverage)
        self.profit_protection_threshold = 2.0 # 2.0% account
        
        # UNIFIED REVERSAL THRESHOLDS - FIXED CONFLICTS
        self.profit_reversal_threshold = 1.0   # 1.0% profit to reverse
        self.loss_reversal_threshold = -1.0    # -1.0% loss to reverse
        self.position_reversal_threshold = -1.5 # -1.5% for other reversals
        
        # SAFE ATR DYNAMIC PROFIT LOCK - UNIFIED
        self.base_profit_lock_threshold = self.profit_lock_threshold  # UNIFIED - no conflicts
        self.atr_multiplier = self.config['atr_multiplier']
        self.min_profit_lock_threshold = 0.5    # 0.5% minimum
        self.max_profit_lock_threshold = 1.5    # 1.5% maximum
        
        # Cooldown
        self.reversal_cooldown_cycles = self.config['signal_cooldown']
        
        # Price-based adjustments
        self.low_price_threshold = 0.05
        self.high_price_threshold = 0.08
        self.low_price_multiplier = 0.9
        self.high_price_multiplier = 0.8
    
    def _load_config(self):
        """Load configuration from JSON"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), 'strategies', 'params_RSI_MFI_Cloud.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Config load error: {e}, using defaults")
            self.config = {
                'symbol': 'ZORA/USDT',
                'oversold_level': 25,
                'overbought_level': 75,
                'signal_cooldown': 2,
                'atr_multiplier': 1.0
            }
    
    @property
    def symbol(self):
        """Get symbol from config"""
        return self.config.get('symbol', 'ZORA/USDT')
    
    @property 
    def rsi_oversold(self):
        """Get RSI oversold from config"""
        return self.config.get('oversold_level', 25)
    
    @property
    def rsi_overbought(self):
        """Get RSI overbought from config"""
        return self.config.get('overbought_level', 75)
    
    @property
    def mfi_oversold(self):
        """Get MFI oversold from config"""
        return self.config.get('oversold_level', 25)
    
    @property
    def mfi_overbought(self):
        """Get MFI overbought from config"""
        return self.config.get('overbought_level', 75)
    
    def calculate_account_pnl_pct(self, unrealized_pnl, account_balance):
        """Calculate P&L as percentage of total account balance"""
        if account_balance <= 0:
            return 0.0
        return (unrealized_pnl / account_balance) * 100
    
    def calculate_position_size(self, balance, price):
        """Calculate SAFE position size"""
        base_value = balance * self.max_position_size
        
        # ZORA price level adjustments
        if price > self.high_price_threshold:
            base_value *= self.high_price_multiplier
        elif price < self.low_price_threshold:
            base_value *= self.low_price_multiplier
        
        return base_value / price
    
    def get_dynamic_profit_lock_threshold(self, atr_pct):
        """Calculate dynamic profit lock threshold based on ATR - UNIFIED"""
        try:
            # Use the SAME base threshold to avoid conflicts
            dynamic_threshold = self.profit_lock_threshold + (atr_pct * self.atr_multiplier * 0.5)
            
            bounded_threshold = max(
                self.min_profit_lock_threshold,
                min(self.max_profit_lock_threshold, dynamic_threshold)
            )
            
            return bounded_threshold
            
        except Exception as e:
            print(f"ATR threshold error: {e}")
            return self.profit_lock_threshold
    
    def should_activate_profit_lock(self, account_pnl_pct, atr_pct=None):
        """Check if profit lock should be activated"""
        if atr_pct is None:
            return account_pnl_pct >= self.profit_lock_threshold
        
        dynamic_threshold = self.get_dynamic_profit_lock_threshold(atr_pct)
        return account_pnl_pct >= dynamic_threshold
    
    def should_take_profit_protection(self, account_pnl_pct):
        """Check if profit protection should trigger"""
        return account_pnl_pct >= self.profit_protection_threshold
    
    def should_reverse_on_signal(self, account_pnl_pct):
        """Check if position should reverse on opposite signal"""
        return account_pnl_pct <= self.position_reversal_threshold
    
    def should_reverse_for_profit(self, account_pnl_pct):
        """Check if should reverse position for profit"""
        return account_pnl_pct >= self.profit_reversal_threshold
    
    def should_reverse_for_loss(self, account_pnl_pct):
        """Check if should reverse position for loss"""
        return account_pnl_pct <= self.loss_reversal_threshold
    
    def is_valid_zora_signal(self, rsi, mfi, trend, volume_ratio=1.0, macd=0, macd_signal=0):
        """ZORA-specific signal validation"""
        
        # Reject extreme levels
        if rsi > 90 and trend != "DOWN":
            return False, "Extreme overbought"
        if rsi < 10 and trend != "UP":
            return False, "Extreme oversold"
        
        return True, "Valid signal"
    
    def get_stop_loss(self, entry_price, side='long'):
        """Calculate stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    # REMOVED: get_take_profit() - Project doesn't use TP, holds until signal/protection
    
    def get_trailing_stop_distance_absolute(self, current_price):
        """Calculate absolute trailing stop distance"""
        return current_price * self.trailing_stop_distance
    
    def get_risk_summary(self, balance, current_price, atr_pct=None):
        """Get SAFE risk summary"""
        position_size = self.calculate_position_size(balance, current_price)
        notional_value = position_size * current_price
        margin_used = notional_value / self.leverage
        
        # ACTUAL RISK CALCULATION
        max_loss_price = notional_value * self.stop_loss_pct  # Price movement loss
        actual_account_risk_pct = (max_loss_price / balance) * 100
        
        return {
            'symbol': self.symbol,
            'balance': balance,
            'position_size': position_size,
            'notional_value': notional_value,
            'margin_used': margin_used,
            'leverage': self.leverage,
            'actual_account_risk_pct': actual_account_risk_pct,  # REAL RISK
            'max_account_loss': max_loss_price,
            'profit_lock_threshold': self.profit_lock_threshold,
            'profit_protection_threshold': self.profit_protection_threshold,
            'profit_reversal_threshold': self.profit_reversal_threshold,
            'loss_reversal_threshold': self.loss_reversal_threshold
        }