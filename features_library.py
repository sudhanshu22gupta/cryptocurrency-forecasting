import pandas as pd
import numpy as np
import math


def daily_returns(ds_price):

    ds_price_t_1 = ds_price.shift(periods=1)
    return (ds_price/ds_price_t_1) - 1


def daily_volatility():

    return


def daily_log_returns(ds_price):

    ds_price_t_1 = ds_price.shift(periods=1)
    return math.log(ds_price/ds_price_t_1)


def daily_squared_returns(ds_price):

    return (daily_returns(ds_price))**2


def realized_volatility(ds_price, N):

    ds_daily_squared_returns = daily_squared_returns(ds_price)
    return ds_daily_squared_returns.rolling(N).sum()

def monthly_realized_volatility(ds_price):
    
    N = 30
    return realized_volatility(ds_price, N)

def weekly_realized_volatility(ds_price):
    
    N = 7
    return realized_volatility(ds_price, N)


def relative_price_range(ds_high, ds_low):

    return 2*((ds_high - ds_low) / (ds_high + ds_low))


def money_flow_index(ds_high, ds_low, ds_close, ds_volume):

    typical_price = (ds_high + ds_low + ds_close) / 3
    money_flow = typical_price * ds_volume
    sum_pos_money_flow = sum(money_flow.loc[money_flow>0])
    sum_neg_money_flow = sum(money_flow.loc[money_flow<0])

    money_ratio = sum_pos_money_flow / sum_neg_money_flow

    return 100 - (100 / (1 + money_ratio))

def avg_directional_movement_index(ds_high, ds_low, ds_close):

    dm_pos = ds_high - ds_high.shift(periods=1)
    dm_neg = ds_low.shift(periods=1) - ds_low
    dm = max([dm_pos, dm_neg, 0])

    # calculate True Range
    tr = max([
        ds_high - ds_low,
        ds_high - ds_close.shift(periods=1),
        ds_close.shift(periods=1) - ds_low,
    ])

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

def accumulation_distribution(ds_close, ds_low, ds_high, ds_volume):

    # cumulative??
    # https://library.tradingtechnologies.com/trade/chrt-ti-accumulation-distribution.html
    return ((ds_close - ds_low) - (ds_high - ds_close) / (ds_high - ds_low)) * ds_volume

def williams_accumulation_distribution(ds_close, ds_low, ds_high, ds_volume):
    
    ds_wad = pd.Series(index=ds_close.index)
    ds_close_n_1 = ds_close.shift(periods=1)

    ds_ad = accumulation_distribution(ds_close, ds_low, ds_high, ds_volume)
    ds_ad_n_1 = ds_ad.shift(periods=1)

    for idx, close_n, close_n_1 in zip(ds_close.index, ds_close, ds_close_n_1):
        if close_n > close_n_1:
            ds_wad[idx] = ds_ad_n_1[idx] + (ds_close[idx] - min([ds_low[idx], close_n_1[idx]]))
        elif close_n > close_n_1:
            ds_wad[idx] = ds_ad_n_1[idx] + (ds_close[idx] - max([ds_high[idx], close_n_1[idx]]))
        else:
            ds_wad[idx] = ds_ad_n_1[idx]
    
    return ds_wad