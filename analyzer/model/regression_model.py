import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


class RegressionModel:

    def __init__(self):
        pass


    def PolynomialFeatures(self, X_train, X_test, y_train, y_test,
                           degree=2, include_bias=False):
        # Трансформируем Х в полиномиальные признаки
        poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_train = pd.DataFrame(poly_features.fit_transform(X_train),
                               columns=poly_features.get_feature_names_out(X_train.columns))
        X_test = pd.DataFrame(poly_features.transform(X_test),
                              columns=poly_features.get_feature_names_out(X_test.columns))
        return X_train, X_test, y_train, y_test

    def OLS(self, X_train, X_test, y_train, y_test,
            prepend=False):
        X_train_plus_const = sm.add_constant(X_train, prepend=prepend)
        X_test_plus_const = sm.add_constant(X_test, prepend=prepend)
        model = OLS(y_train, X_train_plus_const).fit()
        y_pred = pd.DataFrame(model.predict(X_test_plus_const), columns=y_test.columns)
        print(model.summary())
        return y_test, y_pred, model

    def Lasso(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        print('__________')
        print("Коэффициенты Lasso-регрессии:")
        print(pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_}))
        return y_test, y_pred, model

    def Ridge(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        print('__________')
        print("Коэффициенты Ridge-регрессии:")
        print(pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_}))
        return y_test, y_pred, model

    def BayesianRidge(self, X_train, X_test, y_train, y_test):
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.ravel()
        model = BayesianRidge().fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)
        print("Коэффициенты Байесовской регрессии:")
        print(pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_}))
        return y_test, y_pred, model

    def GLM(self, X_train, X_test, y_train, y_test,
            prepend=False, family=sm.families.Gamma(link=sm.families.links.Log())):
        X_train_plus_const = sm.add_constant(X_train, prepend=prepend)
        X_test_plus_const = sm.add_constant(X_test, prepend=prepend)
        model = sm.GLM(y_train, X_train_plus_const, family=family).fit()
        y_pred = pd.DataFrame(model.predict(X_test_plus_const), columns=y_test.columns)
        print(model.summary())
        return y_test, y_pred, model

    def SVR(self, X_train, X_test, y_train, y_test,
            kernel='rbf', C=1.0, epsilon=0.1):
        model = MultiOutputRegressor(SVR(kernel=kernel, C=C, epsilon=epsilon))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model

    def RandomForestRegressor(self, X_train, X_test, y_train, y_test,
                              max_depth=5,
                              min_samples_split=4,
                              min_samples_leaf=2,
                              n_estimators=100):
        model = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model

    def GradientBoostingRegressor(self, X_train, X_test, y_train, y_test):
        model = MultiOutputRegressor(GradientBoostingRegressor())
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model
