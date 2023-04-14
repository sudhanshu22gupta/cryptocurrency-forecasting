from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class BaseReg:

    def __init__(self, scaler=None):
        self.regressor = None
        self.scaler = scaler
        if self.scaler:
            try:
                assert self.scaler in ['StandardScaler']
            except:
                raise NotImplementedError

    def transform_StdScaler(self, X_train):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

    def fit_regressor(self, X_train, Y_train):
        if self.regressor == None:
            self.create_regressor()
        # try:
        #     assert not any(np.isnan(X_train))
        # except:
        #     raise AssertionError(
        #         "RandomForestRegressor cannot handle nan values")
        if self.scaler == 'StandardScaler':
            self.transform_StdScaler(X_train)
        self.regressor.fit(X_train, Y_train)

    def predict_regressor(self, X_test):
        if self.scaler:
            X_test[X_test.columns] = self.scaler.transform(X_test)
        Y_pred = self.regressor.predict(X_test)
        self.Y_pred = pd.Series(Y_pred, index=X_test.index)

    def score_regressor(self, X_test, Y_test):
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        self.score = self.regressor.score(X_test, Y_test)


class RandomForestReg(BaseReg):

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
        self.n_estimators = params["n_estimators"]
        self.criterion = params["criterion"]
        self.max_depth = params["max_depth"]
        self.max_features = params["max_features"]
        # self.min_samples_leaf = params["min_samples_leaf"]

    def create_regressor(self):
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            # min_samples_leaf=self.min_samples_leaf,
        )


class HistGradientBoostingReg(BaseReg):

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
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

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = ElasticNet()


class SGDReg(BaseReg):

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
        self.loss = params["loss"]
        self.max_iter = params["max_iter"]
        self.penalty = params["penalty"]
        self.validation_fraction = params["validation_fraction"]
        self.early_stopping = params["early_stopping"]
        self.n_iter_no_change = params["n_iter_no_change"]
        self.learning_rate = params["learning_rate"]

    def create_regressor(self):
        self.regressor = SGDRegressor(
            loss = self.loss,
            max_iter = self.max_iter,
            penalty = self.penalty,
            validation_fraction = self.validation_fraction,
            early_stopping = self.early_stopping,
            n_iter_no_change = self.n_iter_no_change,
            learning_rate = self.learning_rate,
        )


class TheilSenReg(BaseReg):

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = TheilSenRegressor()


class RANSACReg(BaseReg):

    def __init__(self, scaler=None, **params):
        super().__init__(scaler=scaler)
        # self.loss = params["loss"]
        # self.max_depth = params["max_depth"]
        # self.validation_fraction = params["validation_fraction"]

    def create_regressor(self):
        self.regressor = RANSACRegressor()
