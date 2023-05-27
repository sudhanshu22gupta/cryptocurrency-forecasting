import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class TradingStrategy:

    def __init__(self, ds_daily_log_returns, ds_close):
        self.ds_daily_log_returns = ds_daily_log_returns
        self.ds_close = ds_close

        self.dict_strategies = {
            "SMA_7": self.strategy_simple_moving_average_N(7),
            "SMA_30": self.strategy_simple_moving_average_N(30),
            "SMA_90": self.strategy_simple_moving_average_N(90),
            "SMA_180": self.strategy_simple_moving_average_N(180),

            "EMA_7": self.strategy_exponential_moving_average_N(7),
            "EMA_30": self.strategy_exponential_moving_average_N(30),
            "EMA_90": self.strategy_exponential_moving_average_N(90),
            "EMA_180": self.strategy_exponential_moving_average_N(180),

            "BBS_7": self.strategy_bollinger_bands_N(7),
            "BBS_14": self.strategy_bollinger_bands_N(14),
            "BBS_30": self.strategy_bollinger_bands_N(30),

            "RSI_7": self.strategy_relative_strength_index_N(7),
            "RSI_14": self.strategy_relative_strength_index_N(14),
            "RSI_30": self.strategy_relative_strength_index_N(30),

            "SMA30_RSI14": self.strategy_SMA30_RSI14(),
            "EMA90_RSI14": self.strategy_EMA90_RSI14(),
            "SMA90_RSI14": self.strategy_SMA90_RSI14(),
            "EMA30_RSI14": self.strategy_EMA30_RSI14(),
            "SMA30_BBS7": self.strategy_SMA30_BBS7(),
            "EMA90_BBS7": self.strategy_EMA90_BBS7(),
        }

    def strategy_buy_and_hold(self):

        return (1 + self.ds_daily_log_returns).cumprod()

    def apply_strategy(self, list_strategies, return_payoff=False):
        
        df_signals = pd.DataFrame()
        for strategy in list_strategies:
            assert strategy in self.dict_strategies.keys()
            # apply the strategy
            df_signals[f'signal_{strategy}'] = self.dict_strategies[strategy]
            if return_payoff:
                df_signals[f'payoff_{strategy}'] = self.calc_payoff(df_signals[f'signal_{strategy}'])
        return df_signals
    
    def shift_N_periods(self, ds_timeseries, N):
        return ds_timeseries.shift(periods=N)

    def compute_moving_average_N(self, N):
        return self.ds_close.rolling(N, min_periods=N).mean()

    def compute_exponential_moving_average_N(self, N):
        return self.ds_close.ewm(span=N, adjust=False).mean()

    def compute_STD_N(self, N):
        return self.ds_close.rolling(N, min_periods=N).std()

    def strategy_simple_moving_average_N(self, N):

        ds_moving_avg_N = self.compute_moving_average_N(N)
        ds_moving_avg_N_1 = self.shift_N_periods(ds_moving_avg_N, 1)

        ds_signal = pd.Series([0]*len(ds_moving_avg_N), index=ds_moving_avg_N.index)
        for idx in ds_signal.index:
            if self.ds_close[idx] > ds_moving_avg_N_1[idx]:
                ds_signal[idx] = 1
            elif self.ds_close[idx] < ds_moving_avg_N_1[idx]:
                ds_signal[idx] = -1
        return ds_signal

    def strategy_exponential_moving_average_N(self, N):

        ds_exponential_moving_avg_N = self.compute_exponential_moving_average_N(N)
        ds_exponential_moving_avg_N_1 = self.shift_N_periods(ds_exponential_moving_avg_N, 1)

        ds_signal = pd.Series([0]*len(ds_exponential_moving_avg_N), index=ds_exponential_moving_avg_N.index)
        for idx in ds_signal.index:
            if self.ds_close[idx] > ds_exponential_moving_avg_N_1[idx]:
                ds_signal[idx] = 1
            elif self.ds_close[idx] < ds_exponential_moving_avg_N_1[idx]:
                ds_signal[idx] = -1
        return ds_signal

    def plot_strategy_bollinger_bands_N(self, N):

        ds_moving_avg_N = self.compute_moving_average_N(N)
        ds_STD_N = self.compute_STD_N(N)

        ds_upper_band = ds_moving_avg_N + (2*ds_STD_N)
        ds_lower_band = ds_moving_avg_N - (2*ds_STD_N)

        ds_close_crossing  = self.ds_close.copy()
        ds_close_crossing.loc[~((ds_close_crossing<ds_lower_band) | (ds_close_crossing>ds_upper_band))] = np.nan

        plt.figure(figsize=(12, 6))
        plt.plot(ds_moving_avg_N, ls="--", color="blue", label="Moving Avg Close Price")
        plt.plot(self.ds_close, ls="-", color="blue", label="Close Price")
        plt.plot(ds_close_crossing, color="red", label="Close Price Crossing Bollinger Band")
        plt.fill_between(
            x=ds_close_crossing.index,
            y1=ds_lower_band,
            y2=ds_upper_band,
            alpha=0.3,
            label="Bollinger Band"
        )
        plt.title(f"Trading Strategy: Bollinger Bands (N={N})")
        plt.ylabel("Moving Avg Close Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def strategy_bollinger_bands_N(self, N):

        ds_moving_avg_N = self.compute_moving_average_N(N)
        ds_STD_N = self.compute_STD_N(N)

        ds_upper_band = ds_moving_avg_N + (2*ds_STD_N)
        ds_lower_band = ds_moving_avg_N - (2*ds_STD_N)

        ds_signal = pd.Series([0]*len(ds_moving_avg_N), index=ds_moving_avg_N.index)
        for idx in ds_signal.index:
            if self.ds_close[idx] > ds_upper_band[idx]:
                ds_signal[idx] = -1
            elif self.ds_close[idx] < ds_lower_band[idx]:
                ds_signal[idx] = 1
        return ds_signal

    def _calc_gain_loss_rsi(self, ds_close_rolling):

        ds_close_gain = pd.Series([0]*len(ds_close_rolling), index=ds_close_rolling.index)
        ds_close_loss = pd.Series([0]*len(ds_close_rolling), index=ds_close_rolling.index)

        for idx in ds_close_rolling.index[1:]:
            prev_idx = idx - pd.Timedelta(days=1)
            if ds_close_rolling[idx] > ds_close_rolling[prev_idx]:
                ds_close_gain[idx] = ds_close_rolling[idx] - ds_close_rolling[prev_idx]
            elif ds_close_rolling[idx] < ds_close_rolling[prev_idx]:
                ds_close_loss[idx] = ds_close_rolling[prev_idx] - ds_close_rolling[idx]
        return ds_close_gain.mean() / ds_close_loss.mean()

    def plot_strategy_relative_strength_index_N(self, N):

        ds_rs = self.ds_close.rolling(N, min_periods=N).apply(self._calc_gain_loss_rsi)
        ds_rsi = 100 - (100 / (1 + ds_rs))
        
        RSI_BUY_THR = 30
        RSI_SELL_THR = 70

        ds_rsi_crossing  = ds_rsi.copy()
        ds_rsi_crossing.loc[~((ds_rsi_crossing<RSI_BUY_THR) | (ds_rsi_crossing>RSI_SELL_THR))] = np.nan

        fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 10))
        ax = axs[0]
        ax.plot(self.ds_close, ls="-", color="blue", label="Close Price")
        ax.legend()
        ax.set_ylabel("Close Price")

        ax = axs[1]
        ax.plot(ds_rsi, label="RSI Value")
        ax.plot(ds_rsi_crossing, color="red", label="RSI Value Crossing RSI Bounds")
        ax.axhline(RSI_BUY_THR, ls="--", color="black", label="RSI Buy/Sell Threshold")
        ax.axhline(RSI_SELL_THR, ls="--", color="black")
        ax.set_title(f"Trading Strategy: RSI (N={N})")
        ax.set_ylabel("RSI Value")
        ax.legend()

        plt.suptitle(f"RSI (N={N})")
        plt.tight_layout()
        plt.show()

    def strategy_relative_strength_index_N(self, N):

        ds_rs = self.ds_close.rolling(N, min_periods=N).apply(self._calc_gain_loss_rsi)
        ds_rsi = 100 - (100 / (1 + ds_rs))

        ds_signal = pd.Series([0]*len(ds_rsi), index=ds_rsi.index)
        for idx in ds_signal.index:
            if ds_rsi[idx] < 30:
                ds_signal[idx] = 1
            elif ds_rsi[idx] > 70:
                ds_signal[idx] = -1
        return ds_signal

    def strategy_SMA30_RSI14(self):

        SMA30 = self.strategy_simple_moving_average_N(30)
        RSI14 = self.strategy_relative_strength_index_N(14)
        return (SMA30 + RSI14).apply(np.sign)

    def strategy_EMA90_RSI14(self):

        EMA90 = self.strategy_exponential_moving_average_N(90)
        RSI14 = self.strategy_relative_strength_index_N(14)
        return (EMA90 + RSI14).apply(np.sign)

    def strategy_SMA90_RSI14(self):

        SMA90 = self.strategy_simple_moving_average_N(90)
        RSI14 = self.strategy_relative_strength_index_N(14)
        return (SMA90 + RSI14).apply(np.sign)

    def strategy_EMA30_RSI14(self):

        EMA30 = self.strategy_exponential_moving_average_N(30)
        RSI14 = self.strategy_relative_strength_index_N(14)
        return (EMA30 + RSI14).apply(np.sign)

    def strategy_SMA30_BBS7(self):

        SMA30 = self.strategy_simple_moving_average_N(30)
        BBS7 = self.strategy_bollinger_bands_N(7)
        return (SMA30 + BBS7).apply(np.sign)

    def strategy_EMA90_BBS7(self):

        EMA90 = self.strategy_exponential_moving_average_N(90)
        BBS7 = self.strategy_bollinger_bands_N(7)
        return (EMA90 + BBS7).apply(np.sign)

    def calc_payoff(self, ds_signal):

        return (1 + self.ds_daily_log_returns*ds_signal).cumprod()