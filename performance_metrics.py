import pandas as pd
import numpy as np


class PerformanceMetrics:

    def __init__(self, ds_daily_log_return, df_trading_signals):
        """
        df_trading_signals : dataframe with column names: "signal_{trading_strategy}"
        """
        self.ds_daily_log_return = ds_daily_log_return
        self.df_trading_signals = df_trading_signals

        self.dict_performance_metrics = {
            "cumulative_return": self.cumulative_return,
            "annualized_return": self.annualized_return,
            "average_daily_log_returns": self.average_daily_log_returns,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "annualized_sharpe_ratio": self.annualized_sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
        }

    def compute_performance_metrics(self, performance_metrics):
        
        df_performance_metrics = []
        df_performance_metrics_last = pd.DataFrame(
            index=performance_metrics,
            columns=self.df_trading_signals.columns,
            )
        
        for column in self.df_trading_signals.columns:
            # if column.startswith("singal"):
            trading_strategy = column.replace("singal_", "")
            self.ds_actual_return = self.ds_daily_log_return * self.df_trading_signals[column]
            for metric in performance_metrics:
                ds_perf_metric = self.dict_performance_metrics[metric]()
                ds_perf_metric.name = f"{metric}_{trading_strategy}"
                df_performance_metrics.append(ds_perf_metric)
                df_performance_metrics_last.at[metric, column] = ds_perf_metric[-1]
        df_performance_metrics = pd.concat(df_performance_metrics, axis=1)
        return df_performance_metrics_last, df_performance_metrics

    def cumulative_return(self):

        return (1 + self.ds_actual_return).cumprod() - 1

    def annualized_return(self):

        return ((1 + self.cumulative_return())**(365 / len(self.ds_actual_return))) - 1

    def average_daily_log_returns(self):

        return self.ds_actual_return.expanding().mean()

    def annualized_volatility(self):

        return self.ds_actual_return.expanding().std() * (365)**0.5

    def sharpe_ratio(self):

        return self.average_daily_log_returns() / self.ds_actual_return.expanding().std()

    def annualized_sharpe_ratio(self):

        return self.sharpe_ratio() * (365)**0.5

    def adjusted_sharpe_ratio(self):

        raise NotImplementedError

    def sortino_ratio(self):

        ds_actual_return_neg = self.ds_actual_return.copy()
        ds_actual_return_neg[ds_actual_return_neg > 0] = np.nan
        return self.average_daily_log_returns() / ds_actual_return_neg.expanding().std()

    def max_drawdown(self):

        peak = self.ds_actual_return.expanding().max()
        trough = self.ds_actual_return.expanding().min()
        return (peak - trough) / peak

    def calmar_ratio(self):

        return self.annualized_return() / self.max_drawdown()

