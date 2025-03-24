import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer


class Normalizer:
    def __init__(self):
        pass

    @staticmethod
    def log1p_norm(X_train, X_test, y_train, y_test):
        if np.any(X_train < 0) or np.any(X_test < 0) or np.any(y_train < 0) or np.any(y_test < 0):
            raise ValueError("Все значения в данных должны быть неотрицательными для использования log1p.")
        X_train_normalized = pd.DataFrame(np.log1p(X_train), columns=X_train.columns)
        X_test_normalized = pd.DataFrame(np.log1p(X_test), columns=X_test.columns)
        y_train_normalized = pd.DataFrame(np.log1p(y_train), columns=y_train.columns)
        y_test_normalized = pd.DataFrame(np.log1p(y_test), columns=y_test.columns)
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def MinMaxScaler(train, test, feature_range=(0, 1)):
        scaler = MinMaxScaler(feature_range=feature_range)
        train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return train_normalized, test_normalized, scaler

    @staticmethod
    def StandardScaler(train, test):
        scaler = StandardScaler()
        train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return train_normalized, test_normalized, scaler

    @staticmethod
    def RobustScaler(train, test, quantile_range=(25.0, 75.0)):
        scaler = RobustScaler(quantile_range=quantile_range)
        train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return train_normalized, test_normalized, scaler

    @staticmethod
    def PowerTransformer(train, test, method='yeo-johnson'):
        scaler = PowerTransformer(method=method)
        train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return train_normalized, test_normalized, scaler

    @staticmethod
    def QuantileTransformer(train, test, output_distribution='uniform'):
        scaler = QuantileTransformer(output_distribution=output_distribution)
        train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return train_normalized, test_normalized, scaler

    @staticmethod
    def reset_index(X_train, X_test, y_train, y_test):
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        return X_train, X_test, y_train, y_test