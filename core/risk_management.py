# core/risk_management.py - HFT OPTIMIZED for ZORA

class RiskManager:
    def __init__(self):
        # =============================================
        # HFT-OPTIMIZED RISK MANAGEMENT FOR ZORA
        # =============================================
        
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # HFT-Optimized Leverage & Position Sizing
        self.leverage = 10
        self.max_position_size = 0.05         # 5% per trade (good for HFT)
        self.risk_per_trade = 0.02            # 2% account risk per trade
        
        # HFT-Optimized Price Movement Thresholds
        self.stop_loss_pct = 0.025            # 2.5% price SL
        self.take_profit_pct = 0.05           # 5% price TP
        self.trailing_stop_distance = 0.008  # 0.8% trailing (tighter for HFT)
        
        # HFT-OPTIMIZED Account P&L Thresholds (FASTER PROFITS)
        self.profit_lock_threshold = 0.15     # 0.15% account (DOWN from 0.3%)
        self.profit_protection_threshold = 1.0 # 1.0% account (DOWN from 2.5%)
        self.position_reversal_threshold = -2.0 # -2.0% account (UP from -3.0%)
        
        # HFT-OPTIMIZED ATR DYNAMIC PROFIT LOCK (MORE AGGRESSIVE)
        self.base_profit_lock_threshold = 0.1    # 0.1% base (DOWN from 0.2%)
        self.atr_multiplier = 0.3                # 0.3x ATR (DOWN from 0.4x)
        self.min_profit_lock_threshold = 0.08    # 0.08% minimum (DOWN from 0.15%)
        self.max_profit_lock_threshold = 0.25    # 0.25% maximum (DOWN from 0.6%)
        
        # HFT-OPTIMIZED Cooldown (FASTER REVERSALS)
        self.reversal_cooldown_cycles = 1     # 1 cycle only (DOWN from 2)
        
        # Price-based adjustments (same)
        self.low_price_threshold = 0.05
        self.high_price_threshold = 0.08
        self.low_price_multiplier = 0.9
        self.high_price_multiplier = 0.8
        
        # RSI/MFI Thresholds (same)
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.mfi_oversold = 25
        self.mfi_overbought = 75
    
    def calculate_account_pnl_pct(self, unrealized_pnl, account_balance):
        """Calculate P&L as percentage of total account balance"""
        if account_balance <= 0:
            return 0.0
        return (unrealized_pnl / account_balance) * 100
    
    def calculate_zora_position_size(self, balance, price, volatility_factor=1.0):
        """ZORA-specific position sizing with volatility adjustment"""
        base_value = balance * self.max_position_size
        
        # ZORA-specific price level adjustments
        if price > self.high_price_threshold:
            base_value *= self.high_price_multiplier
        elif price < self.low_price_threshold:
            base_value *= self.low_price_multiplier
        
        # Additional volatility adjustment
        if volatility_factor > 2.0:
            base_value *= 0.8
        
        return base_value / price
    
    def calculate_position_size(self, balance, price):
        """Calculate position size based on ZORA-optimized risk parameters"""
        return self.calculate_zora_position_size(balance, price)
    
    # ==========================================
    # DYNAMIC THRESHOLDS (Single Responsibility)
    # ==========================================
    
    def get_dynamic_profit_lock_threshold(self, atr_pct):
        """Calculate dynamic profit lock threshold based on ATR"""
        try:
            dynamic_threshold = self.base_profit_lock_threshold + (atr_pct * self.atr_multiplier)
            
            bounded_threshold = max(
                self.min_profit_lock_threshold,
                min(self.max_profit_lock_threshold, dynamic_threshold)
            )
            
            return bounded_threshold
            
        except Exception as e:
            print(f"ATR threshold error: {e}")
            return self.profit_lock_threshold
    
    # ==========================================
    # CONDITION CHECKS (Single Responsibility)
    # ==========================================
    
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
    
    # ==========================================
    # SIGNAL VALIDATION (Single Responsibility)
    # ==========================================
    
    def is_valid_zora_signal(self, rsi, mfi, trend, volume_ratio=1.0, macd=0, macd_signal=0):
        """ZORA-specific signal validation - LESS STRICT"""
        
        # Less strict extreme levels (changed from 15/85 to 10/90)
        if rsi > 90 and trend != "DOWN":
            return False, "ZORA extreme overbought"
        if rsi < 10 and trend != "UP":
            return False, "ZORA extreme oversold"
        
        # MACD confirmation for ZORA trades
        if macd != 0 and macd_signal != 0:
            macd_bullish = macd > macd_signal
            if trend == "UP" and not macd_bullish:
                return False, "MACD bearish divergence"
            if trend == "DOWN" and macd_bullish:
                return False, "MACD bullish divergence"
        
        return True, "Valid ZORA signal"
    
    # ==========================================
    # PRICE LEVEL CALCULATIONS (Single Responsibility)
    # ==========================================
    
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
    
    # ==========================================
    # UTILITY METHODS (Single Responsibility)
    # ==========================================
    
    def get_price_zone(self, price):
        """Determine ZORA price zone for risk assessment"""
        if price > self.high_price_threshold:
            return "HIGH_RISK (Above $0.08 resistance)"
        elif price < self.low_price_threshold:
            return "SUPPORT (Below $0.05)"
        else:
            return "NORMAL_RANGE"
    
    # ==========================================
    # COMPREHENSIVE SUMMARY (Single Responsibility)
    # ==========================================
    
    def get_risk_summary(self, balance, current_price, atr_pct=None):
        """Get comprehensive ZORA-optimized risk summary for display"""
        position_size = self.calculate_position_size(balance, current_price)
        notional_value = position_size * current_price
        margin_used = notional_value / self.leverage
        
        # Calculate levels
        sl_long = self.get_stop_loss(current_price, 'long')
        sl_short = self.get_stop_loss(current_price, 'short')
        tp_long = self.get_take_profit(current_price, 'long')
        tp_short = self.get_take_profit(current_price, 'short')
        
        # Get profit lock threshold (dynamic if ATR provided)
        if atr_pct is not None:
            profit_lock_threshold = self.get_dynamic_profit_lock_threshold(atr_pct)
            profit_lock_is_dynamic = True
        else:
            profit_lock_threshold = self.profit_lock_threshold
            profit_lock_is_dynamic = False
        
        return {
            'symbol': self.symbol,
            'optimization': 'ZORA-SPECIFIC',
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
            'profit_lock_threshold': profit_lock_threshold,
            'profit_protection_threshold': self.profit_protection_threshold,
            'position_reversal_threshold': self.position_reversal_threshold,
            'trailing_distance_pct': self.trailing_stop_distance * 100,
            'trailing_distance_abs': self.get_trailing_stop_distance_absolute(current_price),
            'atr_pct': atr_pct or 0,
            'profit_lock_is_dynamic': profit_lock_is_dynamic,
            'rsi_thresholds': f"{self.rsi_oversold}/{self.rsi_overbought}",
            'mfi_thresholds': f"{self.mfi_oversold}/{self.mfi_overbought}",
            'price_zone': self.get_price_zone(current_price)
        }