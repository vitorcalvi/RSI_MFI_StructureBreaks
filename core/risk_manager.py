import json
import os

class RiskManager:
    def __init__(self, config_file="strategies/params_RSI_MFI.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.symbol = "ETHUSDT"
        
        print("⚡ HF Scalping Risk management loaded")
        print(f"💰 Risk per trade: {self.config['fixed_risk_pct']*100}%")
        print(f"🎯 Reward ratio: {self.config['reward_ratio']}:1")
    
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config load error: {e}")
            raise

    
    def calculate_position_size(self, balance, entry_price, stop_price):
        try:
            if balance <= 0 or entry_price <= 0 or stop_price <= 0:
                return 0
            
            risk_amount = balance * self.config['fixed_risk_pct']
            price_diff = abs(entry_price - stop_price)
            if price_diff <= 0:
                return 0
            
            total_fees = self.config['entry_fee_pct'] + self.config['exit_fee_pct']
            position_size = (risk_amount / price_diff) / (1 + total_fees)
            position_size = round(position_size, 3)
            
            if position_size < 0.001:
                return 0
            
            max_size = balance * 0.1 / entry_price
            return min(position_size, max_size)
        except Exception as e:
            print(f"❌ Position size calculation error: {e}")
            return 0
    
    def calculate_take_profit(self, entry_price, stop_loss, side):
        try:
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = risk_distance * self.config['reward_ratio']
            
            if side == "BUY":
                return round(entry_price + reward_distance, 2)
            return round(entry_price - reward_distance, 2)
        except Exception as e:
            print(f"❌ Take profit calculation error: {e}")
            return entry_price
    
    def validate_trade(self, signal, balance, current_price):
        try:
            if balance < 50:
                return False, "Insufficient balance"
            
            stop_distance_pct = abs(current_price - signal['structure_stop']) / current_price
            
            if stop_distance_pct > 0.01:
                return False, "Stop loss too far"
            if stop_distance_pct < 0.0005:
                return False, "Stop loss too close"
            
            potential_profit = stop_distance_pct * self.config['reward_ratio']
            if potential_profit < self.config.get('min_profit_distance', 0.003):
                return False, "Insufficient profit potential"
            
            return True, "Valid trade"
        except Exception as e:
            print(f"❌ Trade validation error: {e}")
            return False, "Validation error"
    
    def should_close_position(self, current_price, entry_price, side, unrealized_pnl, position_age_seconds):
        try:
            if position_age_seconds > self.config.get('max_position_time', 120):
                return True, "Max hold time exceeded"
            
            position_value = abs(entry_price)
            pnl_pct = abs(unrealized_pnl) / position_value if position_value > 0 else 0
            
            if unrealized_pnl < 0 and pnl_pct > self.config.get('emergency_stop_pct', 0.02):
                return True, "Emergency stop loss triggered"
            
            if unrealized_pnl > 0 and pnl_pct > 0.005:
                return True, "Quick profit target reached"
            
            return False, "Continue holding"
        except Exception:
            return False, "Error in position check"
    
    def get_trailing_stop_distance(self):
        return self.config['trailing_stop_pct']
    
    def calculate_breakeven_price(self, entry_price, quantity, side):
        try:
            position_value = quantity * entry_price
            total_fees = position_value * (self.config['entry_fee_pct'] + self.config['exit_fee_pct'])
            fee_per_unit = total_fees / quantity
            
            if side.upper() == "BUY":
                return round(entry_price + fee_per_unit, 2)
            return round(entry_price - fee_per_unit, 2)
        except Exception:
            return entry_price
    
    def adjust_risk_for_volatility(self, price_data):
        try:
            if len(price_data) < 10:
                return self.config['fixed_risk_pct']
            
            recent_data = price_data.tail(10)
            high_low_diff = (recent_data['high'] - recent_data['low']) / recent_data['close']
            avg_volatility = high_low_diff.mean()
            
            base_risk = self.config['fixed_risk_pct']
            
            if avg_volatility > 0.015:
                adjusted_risk = base_risk * 0.5
            elif avg_volatility < 0.005:
                adjusted_risk = base_risk * 1.5
            else:
                adjusted_risk = base_risk
            
            return max(0.002, min(0.01, adjusted_risk))
        except Exception:
            return self.config['fixed_risk_pct']
    
    def get_max_position_size(self, balance, price):
        try:
            max_position_value = balance * 0.1
            return round(max_position_value / price, 3)
        except Exception:
            return 0
    
    def calculate_fees(self, quantity, price):
        try:
            position_value = quantity * price
            return position_value * (self.config['entry_fee_pct'] + self.config['exit_fee_pct'])
        except Exception:
            return 0
    
    def save_config(self):
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Config save error: {e}")
            return False