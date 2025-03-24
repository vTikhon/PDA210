import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from analyzer.model import RegressionModel
from data import DataPreparation
from data.io import Reader
from data.normalizer import Normalizer


class LaunchPredict:

    def launch_predict(self, data):
        X_new = DataPreparation.create_dataframe(data)
        csv_path = Reader.get_csv_path('notebooks\dataset', 'RK_554B_data.csv')
        df_original = Reader.read_csv(csv_path)
        df = df_original.copy().reset_index(drop=True)
        df = df.drop('number', axis=1)
        mask = df['resistor'] != 6000
        df = df[mask]

        # подготавливаем данные
        target = ['resistor']
        exclude_features = ['resistor', 'freq', 'r', 'c0', 'cq', 'q1000', 'rc/rb']
        X_train, X_test, y_train, y_test = DataPreparation().train_test_split(df, exclude_features, target)

        # проводим нормализацию
        X_train, X_test, scaler_x = Normalizer().MinMaxScaler(X_train, X_test)
        y_train, y_test, scaler_y = Normalizer().MinMaxScaler(y_train, y_test)
        X_train, X_test, y_train, y_test = Normalizer().reset_index(X_train, X_test, y_train, y_test)

        # Применим метод опорных векторов
        y_test, y_pred, model = RegressionModel().SVR(X_train, X_test, y_train, y_test, kernel='rbf', C=1.0,
                                                      epsilon=0.01)

        # Производим расчёт y_pred от поступившего X_new
        y_test = pd.DataFrame(scaler_y.inverse_transform(y_test), columns=y_test.columns)
        y_pred = pd.DataFrame(scaler_y.inverse_transform(y_pred), columns=y_pred.columns)
        R_square_metric = r2_score(y_test, y_pred)
        MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        X_new = pd.DataFrame(scaler_x.transform(X_new), columns=X_new.columns)
        y_pred = pd.DataFrame(model.predict(X_new), columns=y_pred.columns)
        y_pred = pd.DataFrame(scaler_y.inverse_transform(y_pred), columns=y_pred.columns)
        return (f"Resistor value = {y_pred.iloc[0, 0]:.0f} Ohms\n\n"
                f"Applied Support Vector Regression Method\n"
                f"R-square: {R_square_metric:.2f}\n"
                f"Mean Absolute Error = {MAPE:.2f}")





