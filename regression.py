from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, TheilSenRegressor, RANSACRegressor
import numpy as np
import pandas as pd


class BaseReg:

    def __init__(self):
        self.regressor = None

    def fit_regressor(self, X_train, Y_train):
        if self.regressor == None:
            self.create_regressor()
        # try:
        #     assert not any(np.isnan(X_train))
        # except:
        #     raise AssertionError(
        #         "RandomForestRegressor cannot handle nan values")
        self.regressor.fit(X_train, Y_train)

    def predict_regressor(self, X_test):
        Y_pred = self.regressor.predict(X_test)
        self.Y_pred = pd.Series(Y_pred, index=X_test.index)

    def score_regressor(self, X_test, Y_test):
        self.score = self.regressor.score(X_test, Y_test)


class RandomForestReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        self.n_estimators = params["n_estimators"]
        self.criterion = params["criterion"]
        self.max_depth = params["max_depth"]
        self.max_features = params["max_features"]

    def create_regressor(self):
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
        )


class HistGradientBoostingReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        self.loss = params["loss"]
        self.max_depth = params["max_depth"]
        self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = HistGradientBoostingRegressor(
            loss=self.loss,
            max_depth=self.max_depth,
            validation_fraction=self.validation_fraction,
        )


class ElasticNetReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = ElasticNet()


class SGDReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = SGDRegressor()


class TheilSenReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = TheilSenRegressor()


class RANSACReg(BaseReg):

    def __init__(self, **params):
        super().__init__()
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = RANSACRegressor()
