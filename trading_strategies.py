import pandas as pd

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

    def compute_moving_average_N(self, N):

        self.ds_moving_avg_N = self.ds_close.rolling(N, min_periods=N).mean()
        self.ds_moving_avg_N_1 = self.ds_moving_avg_N.shift(periods=1)

    def compute_exponential_moving_average_N(self, N):

        self.ds_exponential_moving_avg_N = self.ds_close.ewm(span=N, adjust=False).mean()
        self.ds_exponential_moving_avg_N_1 = self.ds_exponential_moving_avg_N.shift(periods=1)

    def compute_STD_N(self, N):

        self.ds_STD_N = self.ds_close.rolling(N, min_periods=N).std()
        self.ds_STD_N_1 = self.ds_STD_N.shift(periods=1)

    def strategy_simple_moving_average_N(self, N):

        self.compute_moving_average_N(N)

        ds_signal = pd.Series([0]*len(self.ds_moving_avg_N), index=self.ds_moving_avg_N.index)
        for idx in ds_signal.index:
            if self.ds_close[idx] > self.ds_moving_avg_N_1[idx]:
                ds_signal[idx] = 1
            elif self.ds_close[idx] < self.ds_moving_avg_N_1[idx]:
                ds_signal[idx] = -1
        return ds_signal

    def strategy_exponential_moving_average_N(self, N):

        self.compute_exponential_moving_average_N(N)

        ds_signal = pd.Series([0]*len(self.ds_exponential_moving_avg_N), index=self.ds_exponential_moving_avg_N.index)
        for idx in ds_signal.index:
            if self.ds_close[idx] > self.ds_exponential_moving_avg_N_1[idx]:
                ds_signal[idx] = 1
            elif self.ds_close[idx] < self.ds_exponential_moving_avg_N_1[idx]:
                ds_signal[idx] = -1
        return ds_signal

    def strategy_bollinger_bands_N(self, N):

        self.compute_moving_average_N(N)
        self.compute_STD_N(N)

        ds_upper_band = self.ds_moving_avg_N + (2*self.ds_STD_N)
        ds_lower_band = self.ds_moving_avg_N - (2*self.ds_STD_N)

        ds_signal = pd.Series([0]*len(self.ds_moving_avg_N), index=self.ds_moving_avg_N.index)
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

    def calc_payoff(self, ds_signal):

        return (1 + self.ds_daily_log_returns*ds_signal).cumprod()