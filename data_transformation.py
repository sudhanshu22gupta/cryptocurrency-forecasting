import pandas as pd

class CleanData:

    def __init__(self, df):
        self.df = df

    def make_datetime_index(self, date_col='Date'):
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df.index = self.df[date_col]
        self.df.index.name = None
        self.df.drop(columns=[date_col], inplace=True)

    def resample(self, freq='1D'):
        self.df = self.df.resample(freq).first()
    
    def strip_column_name(self):
        self.df.rename(
            columns={col: col.strip() for col in self.df.columns}, 
            inplace=True,
            )

    def ffill(self):
        self.df.ffill()
