import pandas as pd
import numpy as np
import math


class FeatureTransformations:

    def __init__(self, df_asset):

        self.df_asset = df_asset

        self.ds_open = self.df_asset['Open']
        self.ds_high = self.df_asset['High']
        self.ds_low = self.df_asset['Low']
        self.ds_close = self.df_asset['Close']
        self.ds_volume = self.df_asset['Volume']

        self.ds_close_t_1 = self.ds_close.shift(periods=1)
        self.ds_high_t_1 = self.ds_high.shift(periods=1)
        self.ds_low_t_1 = self.ds_low.shift(periods=1)
    
    def transform_assets(self):
        
        self.df_asset['daily_returns'] = self.daily_returns()
        self.df_asset['daily_volatility'] = self.daily_volatility()
        self.df_asset['daily_log_returns'] = self.daily_log_returns()
        self.df_asset['daily_squared_returns'] = self.daily_squared_returns()
        self.df_asset['monthly_realized_volatility'] = self.monthly_realized_volatility()
        self.df_asset['weekly_realized_volatility'] = self.weekly_realized_volatility()
        self.df_asset['relative_price_range'] = self.relative_price_range()
        self.df_asset['money_flow_index'] = self.money_flow_index()
        self.df_asset['avg_directional_movement_index'] = self.avg_directional_movement_index()
        self.df_asset['williams_accumulation_distribution'] = self.williams_accumulation_distribution()
        self.df_asset['on_balance_volume'] = self.on_balance_volume()
        self.df_asset['pma_10'] = self.pma_10()
        self.df_asset['pma_30'] = self.pma_30()
        self.df_asset['pma_60'] = self.pma_60()

    def transform_snp500(self):

        self.df_asset['daily_returns'] = self.daily_returns()
        self.df_asset['daily_log_returns'] = self.daily_log_returns()
        self.df_asset['monthly_realized_volatility'] = self.monthly_realized_volatility()
        self.df_asset['pma_10'] = self.pma_10()
        self.df_asset['pma_30'] = self.pma_30()
        self.df_asset['pma_60'] = self.pma_60()

    def daily_returns(self):

        return (self.ds_close/self.ds_close_t_1) - 1

    def daily_volatility(self):

        return self.ds_close.expanding().std()

    def daily_log_returns(self):

        return np.log(self.ds_close/self.ds_close_t_1)

    def daily_squared_returns(self):

        return (self.daily_log_returns())**2

    def realized_volatility(self, N):

        ds_daily_squared_returns = self.daily_squared_returns()
        return np.sqrt(ds_daily_squared_returns.rolling(N, min_periods=N).sum())

    def monthly_realized_volatility(self):

        return self.realized_volatility(N=30)

    def weekly_realized_volatility(self):

        return self.realized_volatility(N=7)

    def relative_price_range(self):

        return 2*((self.ds_high - self.ds_low) / (self.ds_high + self.ds_low))

    def money_flow_index(self):

        typical_price = (self.ds_high + self.ds_low + self.ds_close) / 3
        money_flow = typical_price * self.ds_volume
        typical_price_t_1 = typical_price.shift(periods=1)
        sign_money_flow = typical_price - typical_price_t_1

        pos_money_flow = money_flow.copy()
        pos_money_flow.loc[sign_money_flow < 0] = 0
        sum_pos_money_flow = pos_money_flow.expanding().sum()

        neg_money_flow = money_flow.copy()
        neg_money_flow.loc[sign_money_flow > 0] = 0
        sum_neg_money_flow = neg_money_flow.expanding().sum()

        money_ratio = sum_pos_money_flow / sum_neg_money_flow

        return 100 - (100 / (1 + money_ratio))

    def avg_directional_movement_index(self):

        dm_pos = self.ds_high - self.ds_high_t_1
        dm_neg = self.ds_low_t_1 - self.ds_low
        # To-Do: Check
        # dm = max([dm_pos, dm_neg, 0])

        # calculate True Range
        tr = pd.DataFrame(np.array([
            self.ds_high - self.ds_low,
            self.ds_high - self.ds_close_t_1,
            self.ds_close_t_1 - self.ds_low,
        ]).T, index=self.ds_high.index).apply(max, axis=1)

        # To-Do: Confirm implementation w/ Rastogi
        period = 14
        dm14_pos = dm_pos.ewm(alpha=1.0 / period, adjust=False).mean()
        dm14_neg = dm_neg.ewm(alpha=1.0 / period, adjust=False).mean()
        tr14 = tr.ewm(alpha=1.0 / period, adjust=False).mean()

        di14_pos = dm14_pos / tr14
        di14_neg = dm14_neg / tr14

        di_diff = abs(di14_pos - di14_neg)

        # directional index
        dx = di_diff / (abs(di14_pos) + abs(di14_neg))

        # average directional index
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

        return adx

    def accumulation_distribution(self):
        """
        DEPRECATED
        """
        raise NotImplementedError()
        # https://library.tradingtechnologies.com/trade/chrt-ti-accumulation-distribution.html
        return (((self.ds_close - self.ds_low) - (self.ds_high - self.ds_close) / (self.ds_high - self.ds_low)) * self.ds_volume).cumsum()

    def williams_accumulation_distribution(self):

        ds_wad = pd.Series(np.zeros(shape=self.ds_close.shape), index=self.ds_close.index)
        
        for idx, close_n, close_n_1 in zip(self.ds_close.index[1:], self.ds_close[1:], self.ds_close_t_1[1:]):
            if close_n > close_n_1:
                ds_wad[idx] = ds_wad[idx-pd.Timedelta(days=1)] + (
                    self.ds_close[idx] - min([self.ds_low[idx], close_n_1]))
            elif close_n < close_n_1:
                ds_wad[idx] = ds_wad[idx-pd.Timedelta(days=1)] + (
                    self.ds_close[idx] - max([self.ds_high[idx], close_n_1]))
            else:
                ds_wad[idx] = ds_wad[idx-pd.Timedelta(days=1)]
        # ds_wad.iloc[0] = np.nan
        return ds_wad

    def price_to_moving_avg_ratio_i(self, window_size):

        return self.ds_close / self.ds_close.rolling(window_size, min_periods=window_size).mean()

    def pma_10(self):

        return self.price_to_moving_avg_ratio_i(window_size=10)

    def pma_30(self):

        return self.price_to_moving_avg_ratio_i(window_size=30)

    def pma_60(self):

        return self.price_to_moving_avg_ratio_i(window_size=60)

    def on_balance_volume(self):

        ds_obv = pd.Series(np.zeros(shape=self.ds_close.shape), index=self.ds_close.index)

        for idx, close_n, close_n_1, volume_n in zip(self.ds_close.index[1:], self.ds_close[1:], self.ds_close_t_1[1:], self.ds_volume[1:]):

            if close_n > close_n_1:
                ds_obv[idx] = ds_obv[idx-pd.Timedelta(days=1)] + volume_n
            elif close_n < close_n_1:
                ds_obv[idx] = ds_obv[idx-pd.Timedelta(days=1)] - volume_n
            else:
                ds_obv[idx] = ds_obv[idx-pd.Timedelta(days=1)]
        
        return ds_obv