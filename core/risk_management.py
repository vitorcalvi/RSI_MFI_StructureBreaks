class RiskManager:
    def __init__(self):
        # =============================================
        # ZORA-OPTIMIZED RISK MANAGEMENT PARAMETERS
        # =============================================
        
        # Trading Symbol
        self.symbol = "ZORA/USDT"
        
        # ZORA-Optimized Leverage & Position Sizing
        self.leverage = 10                    # 10x leverage
        self.max_position_size = 0.05         # 5% of balance per trade (REDUCED FOR ZORA)
        self.risk_per_trade = 0.02            # 2% account risk per trade (REDUCED FOR CRYPTO)
        
        # ZORA-Optimized Price Movement Thresholds (as percentages)
        self.stop_loss_pct = 0.025            # 2.5% price movement = stop loss (TIGHTER FOR CRYPTO)
        self.take_profit_pct = 0.05           # 5% price movement = take profit (EARLIER FOR CRYPTO)
        self.break_even_pct = 0.008           # 0.8% price movement = profit lock trigger
        self.trailing_stop_distance = 0.006  # 0.6% trailing distance (TIGHTER)
        
        # ZORA-Optimized Account P&L Thresholds (as percentages of total account)
        self.profit_lock_threshold = 0.3      # 0.3% account P&L = activate trailing stop (EARLIER)
        self.profit_protection_threshold = 2.5 # 2.5% account P&L = take profit & cooldown (EARLIER)
        self.loss_switch_threshold = -6.0     # -6% account P&L = reverse position (LESS AGGRESSIVE)
        self.position_reversal_threshold = -3.0 # -3% account P&L = reverse on signal (QUICKER)
        
        # ZORA-Optimized ATR DYNAMIC PROFIT LOCK
        self.base_profit_lock_threshold = 0.2   # 0.2% base threshold (MORE CONSERVATIVE)
        self.atr_multiplier = 0.4               # 0.4x ATR sensitivity (MORE CONSERVATIVE)
        self.min_profit_lock_threshold = 0.15   # 0.15% minimum (REDUCED)
        self.max_profit_lock_threshold = 0.6    # 0.6% maximum (REDUCED)
        
        # ZORA-Optimized Cooldown & Control
        self.reversal_cooldown_cycles = 2     # 2 cycles cooldown (FASTER FOR CRYPTO)
        
        # ZORA-Optimized Price-based Position Adjustments
        self.low_price_threshold = 0.05       # Below $0.05 = reduce position 10%
        self.high_price_threshold = 0.08      # Above $0.08 = reduce position 20% (ZORA RESISTANCE)
        self.low_price_multiplier = 0.9       # 90% of normal size for low prices
        self.high_price_multiplier = 0.8      # 80% of normal size for high prices (RISK REDUCTION)
        
        # ZORA-Optimized Ranging Market Exit Parameters
        self.ranging_exit_cycles = 3          # Exit after 3 ranging cycles (FASTER)
        self.ranging_profit_threshold = 0.15  # Take profit at 0.15% in ranging (SMALLER PROFITS)
        self.ranging_loss_threshold = -0.6    # Cut loss at -0.6% in ranging (FASTER CUTS)
        
        # ZORA-Specific RSI/MFI Thresholds (MISSING IN YOUR FILE)
        self.rsi_oversold = 25                # More strict than 30
        self.rsi_overbought = 75              # More strict than 70  
        self.mfi_oversold = 25                # More strict
        self.mfi_overbought = 75              # More strict
    
    def calculate_account_pnl_pct(self, unrealized_pnl, account_balance):
        """Calculate P&L as percentage of total account balance"""
        if account_balance <= 0:
            return 0.0
        return (unrealized_pnl / account_balance) * 100
    
    def calculate_zora_position_size(self, balance, price, volatility_factor=1.0):
        """ZORA-specific position sizing with volatility adjustment (MISSING IN YOUR FILE)"""
        base_value = balance * self.max_position_size
        
        # ZORA-specific price level adjustments
        if price > self.high_price_threshold:  # Above $0.08 resistance
            base_value *= self.high_price_multiplier  # Reduce to 80%
        elif price < self.low_price_threshold:  # Below $0.05 support
            base_value *= self.low_price_multiplier   # Reduce to 90%
        
        # Additional volatility adjustment for ZORA
        if volatility_factor > 2.0:  # High volatility period
            base_value *= 0.8  # Further reduce position size
        
        return base_value / price
    
    def calculate_position_size(self, balance, price):
        """Calculate position size based on ZORA-optimized risk parameters"""
        # Use ZORA-specific position sizing
        return self.calculate_zora_position_size(balance, price)
    
    def get_dynamic_profit_lock_threshold(self, atr_pct):
        """Calculate dynamic profit lock threshold based on ATR (FIXED CALCULATION)"""
        try:
            # FIXED Formula: base + (atr_pct * multiplier)
            dynamic_threshold = self.base_profit_lock_threshold + (atr_pct * self.atr_multiplier)
            
            # Apply bounds
            bounded_threshold = max(
                self.min_profit_lock_threshold,
                min(self.max_profit_lock_threshold, dynamic_threshold)
            )
            
            return bounded_threshold
            
        except Exception as e:
            print(f"ATR threshold error: {e}")
            return self.profit_lock_threshold  # Fallback to static
    
    def should_activate_profit_lock(self, account_pnl_pct, atr_pct=None):
        """Check if profit lock should be activated (UPDATED - ATR OPTIONAL)"""
        if atr_pct is None:
            # Fallback to static threshold
            return account_pnl_pct >= self.profit_lock_threshold
        
        # Use dynamic threshold
        dynamic_threshold = self.get_dynamic_profit_lock_threshold(atr_pct)
        return account_pnl_pct >= dynamic_threshold
    
    def should_take_profit_protection(self, account_pnl_pct):
        """Check if profit protection should trigger"""
        return account_pnl_pct >= self.profit_protection_threshold
    
    def should_switch_position(self, account_pnl_pct):
        """Check if position should be switched due to loss"""
        return account_pnl_pct <= self.loss_switch_threshold
    
    def should_reverse_on_signal(self, account_pnl_pct):
        """Check if position should reverse on opposite signal"""
        return account_pnl_pct <= self.position_reversal_threshold
    
    def should_exit_ranging_market(self, pnl_pct, current_trend, current_rsi, current_mfi, position_side, ranging_cycles):
        """Check if position should exit due to ranging market conditions (ZORA-OPTIMIZED)"""
        # Exit after too many ranging cycles (faster for ZORA)
        if ranging_cycles >= self.ranging_exit_cycles:
            return True, f"ZORA ranging market exit ({ranging_cycles} cycles)"
        
        # Take smaller profits in ranging market (optimized for crypto volatility)
        if current_trend == "SIDEWAYS" and pnl_pct >= self.ranging_profit_threshold:
            return True, f"ZORA ranging profit exit ({pnl_pct:.2f}%)"
        
        # Cut losses faster in ranging market (crypto-specific)
        if current_trend == "SIDEWAYS" and pnl_pct <= self.ranging_loss_threshold:
            return True, f"ZORA ranging loss exit ({pnl_pct:.2f}%)"
        
        # Exit on extreme RSI/MFI in ranging (ZORA-specific thresholds)
        if current_trend == "SIDEWAYS":
            if position_side == "Buy" and current_rsi >= self.rsi_overbought and current_mfi >= self.mfi_overbought:
                return True, "ZORA ranging overbought exit"
            elif position_side == "Sell" and current_rsi <= self.rsi_oversold and current_mfi <= self.mfi_oversold:
                return True, "ZORA ranging oversold exit"
        
        return False, ""
    
    def is_valid_zora_signal(self, rsi, mfi, trend, volume_ratio=1.0, macd=0, macd_signal=0):
        """ZORA-specific signal validation - VOLUME CHECK REMOVED"""
        
        # Avoid extreme overbought/oversold without trend confirmation
        if rsi > 85 and trend != "DOWN":
            return False, "ZORA extreme overbought"
        if rsi < 15 and trend != "UP":
            return False, "ZORA extreme oversold"
        
        # MACD confirmation for ZORA trades
        if macd != 0 and macd_signal != 0:
            macd_bullish = macd > macd_signal
            if trend == "UP" and not macd_bullish:
                return False, "MACD bearish divergence"
            if trend == "DOWN" and macd_bullish:
                return False, "MACD bullish divergence"
            
        return True, "Valid ZORA signal"

    
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
    
    def get_price_zone(self, price):
        """Determine ZORA price zone for risk assessment (MISSING IN YOUR FILE)"""
        if price > self.high_price_threshold:
            return "HIGH_RISK (Above $0.08 resistance)"
        elif price < self.low_price_threshold:
            return "SUPPORT (Below $0.05)"
        else:
            return "NORMAL_RANGE"
    
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
            'loss_switch_threshold': self.loss_switch_threshold,
            'trailing_distance_pct': self.trailing_stop_distance * 100,
            'trailing_distance_abs': self.get_trailing_stop_distance_absolute(current_price),
            'atr_pct': atr_pct or 0,
            'profit_lock_is_dynamic': profit_lock_is_dynamic,
            'rsi_thresholds': f"{self.rsi_oversold}/{self.rsi_overbought}",
            'mfi_thresholds': f"{self.mfi_oversold}/{self.mfi_overbought}",
            'ranging_cycles_limit': self.ranging_exit_cycles,
            'price_zone': self.get_price_zone(current_price)
        }