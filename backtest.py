import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class EthHFTBacktester:
    def __init__(self, data_path: str, params: dict, initial_balance: float = 10000.0):
        # Backtest parameters
        self.timeframe = '1min'
        self.data_path = data_path
        self.initial_balance = initial_balance

        # Strategy parameters (tuned)
        self.rsi_length = params['rsi_length']
        self.mfi_length = params['mfi_length']
        self.oversold = params['oversold']
        self.overbought = params['overbought']
        self.structure_lookback = params['structure_lookback']
        self.structure_buffer_pct = params['structure_buffer_pct']

        # Risk management
        self.fixed_risk_pct = params['fixed_risk_pct']
        self.trailing_stop_pct = params['trailing_stop_pct']
        self.profit_lock_threshold = params['profit_lock_threshold']
        self.reward_ratio = params['reward_ratio']

        # Trading costs
        self.entry_fee_pct = params['entry_fee_pct']
        self.exit_fee_pct = params['exit_fee_pct']

        # Internal state
        self.data = None
        self.trades = []
        self.balance = initial_balance

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'], index_col='timestamp')
        df = df.resample(self.timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        self.data = df

    def compute_indicators(self):
        close = self.data['close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.data['rsi'] = 100 - 100/(1+rs)

        tp = (self.data['high'] + self.data['low'] + close) / 3
        mf = tp * self.data['volume']
        sign = tp.diff()
        pos_mf = mf.where(sign > 0, 0)
        neg_mf = mf.where(sign <= 0, 0)
        pos_ema = pos_mf.ewm(alpha=1/self.mfi_length, adjust=False).mean()
        neg_ema = neg_mf.ewm(alpha=1/self.mfi_length, adjust=False).mean()
        mf_ratio = pos_ema / neg_ema.replace(0, np.nan)
        self.data['mfi'] = 100 - 100/(1+mf_ratio)

        self.data[['rsi', 'mfi']] = self.data[['rsi', 'mfi']].fillna(50)

    def run_backtest(self):
        self.load_data()
        self.compute_indicators()
        self.trades.clear()
        self.balance = self.initial_balance
        position = None

        for ts, row in self.data.iterrows():
            price = row['close']
            if position is None:
                if row['rsi'] < self.oversold and row['mfi'] < self.oversold:
                    position = self.open_position('BUY', price, ts)
                elif row['rsi'] > self.overbought and row['mfi'] > self.overbought:
                    position = self.open_position('SELL', price, ts)
            else:
                if self.manage_position(position, price, ts):
                    position = None

        if position:
            self.close_position(position, self.data['close'].iloc[-1], self.data.index[-1], 'End')

        return self.evaluate()

    def open_position(self, side, price, ts):
        risk_amount = self.balance * self.fixed_risk_pct
        window = self.data.loc[:ts].iloc[-self.structure_lookback:]
        if side == 'BUY':
            stop_level = window['low'].min() - price * (self.structure_buffer_pct / 100)
        else:
            stop_level = window['high'].max() + price * (self.structure_buffer_pct / 100)
        size = risk_amount / abs(price - stop_level)
        tp = price + (price - stop_level) * self.reward_ratio if side == 'BUY' else price - (stop_level - price) * self.reward_ratio

        entry_cost = price * size * self.entry_fee_pct
        self.balance -= entry_cost

        return {
            'side': side,
            'entry': price,
            'stop': stop_level,
            'tp': tp,
            'size': size,
            'open_ts': ts
        }

    def manage_position(self, pos, price, ts):
        pnl_pct = ((price - pos['entry']) / pos['entry'] * 100) if pos['side'] == 'BUY' else ((pos['entry'] - price) / pos['entry'] * 100)
        if (pos['side'] == 'BUY' and price <= pos['stop']) or (pos['side'] == 'SELL' and price >= pos['stop']):
            self.close_position(pos, price, ts, 'Stop Loss')
            return True
        if (pos['side'] == 'BUY' and price >= pos['tp']) or (pos['side'] == 'SELL' and price <= pos['tp']):
            self.close_position(pos, price, ts, 'Take Profit')
            return True
        if pnl_pct >= self.profit_lock_threshold:
            new_stop = price * (1 - self.trailing_stop_pct) if pos['side'] == 'BUY' else price * (1 + self.trailing_stop_pct)
            pos['stop'] = max(pos['stop'], new_stop) if pos['side'] == 'BUY' else min(pos['stop'], new_stop)
        return False

    def close_position(self, pos, price, ts, reason):
        size = pos['size']
        pnl = (price - pos['entry']) * size if pos['side'] == 'BUY' else (pos['entry'] - price) * size
        exit_cost = price * size * self.exit_fee_pct
        net_pnl = pnl - exit_cost
        self.balance += net_pnl
        self.trades.append({'pnl': net_pnl})

    def evaluate(self):
        df = pd.DataFrame(self.trades)
        total_pnl = df['pnl'].sum()
        return total_pnl / self.initial_balance * 100

    def report(self):
        # Summary report
        if not self.trades:
            print("No trades executed.")
            return
        df = pd.DataFrame(self.trades)
        total_ret = df['pnl'].sum() / self.initial_balance * 100
        win_rate = len(df[df['pnl'] > 0]) / len(df) * 100
        print(f"Final Balance: {self.balance:.2f}")
        print(f"Total Return: {total_ret:.2f}% | Win Rate: {win_rate:.1f}% | Trades: {len(df)}")

# Randomized tuner over multiple parameters
def tune(data_path, max_trials=200, seed=42):
    import random
    random.seed(seed)
    grid = {
        'rsi_length': [3,5,7],
        'mfi_length': [3,5,7],
        'oversold': [30,35,40,45],
        'overbought': [55,60,65,70],
        'structure_lookback': [20,60,120],
        'structure_buffer_pct': [0.1,0.2,0.3],
        'fixed_risk_pct': [0.005,0.01,0.02],
        'trailing_stop_pct': [0.005,0.008,0.01],
        'profit_lock_threshold': [2.0,3.0,4.0],
        'reward_ratio': [3.0,4.0,5.0]
    }
    keys = list(grid.keys())
    best = {'params': None, 'return': -np.inf}

    for _ in range(max_trials):
        params = {k: random.choice(grid[k]) for k in keys}
        params['entry_fee_pct'] = 0.00055
        params['exit_fee_pct'] = 0.00055
        bt = EthHFTBacktester(data_path, params)
        ret = bt.run_backtest()
        if ret > best['return']:
            best = {'params': params.copy(), 'return': ret}
    return best

if __name__ == '__main__':
    # Tune parameters
    best = tune('_data/ETHUSDT_1_7d_20250731_071705.csv', max_trials=200)
    print(f"Best params: {best['params']} → Return: {best['return']:.2f}%")

    # Run full backtest with tuned parameters and detailed report
    print("\nRunning full backtest with tuned parameters...\n")
    bt = EthHFTBacktester('data/ETHUSDT_1_7d_20250731_071705.csv', best['params'])
    bt.load_data()
    bt.compute_indicators()
    bt.trades.clear()
    bt.balance = bt.initial_balance
    position = None
    for ts, row in bt.data.iterrows():
        price = row['close']
        if position is None:
            if row['rsi'] < bt.oversold and row['mfi'] < bt.oversold:
                position = bt.open_position('BUY', price, ts)
            elif row['rsi'] > bt.overbought and row['mfi'] > bt.overbought:
                position = bt.open_position('SELL', price, ts)
        else:
            if bt.manage_position(position, price, ts):
                position = None
    if position:
        bt.close_position(position, bt.data['close'].iloc[-1], bt.data.index[-1], 'End')

    # Final summary
    bt.report()

    # Plot equity curve
    eq = [bt.initial_balance]
    times = [bt.data.index[0]]
    for trade in bt.trades:
        eq.append(eq[-1] + trade['pnl'])
        times.append(trade['close_ts'])
    plt.figure(figsize=(10,5))
    plt.plot(times, eq)
    plt.title('Equity Curve with Tuned Strategy')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.show()
