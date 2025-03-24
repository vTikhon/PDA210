import numpy as np
import pandas as pd


class LinearRegressionCustom:
    def __init__(self):
        self.w = None

    def _normal_equation(self, X, y):
        cond = np.linalg.cond(X)
        if cond < 10:
            self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        elif 10 <= cond < 1000:
            Q, R = np.linalg.qr(X)
            self.w = np.linalg.inv(R).dot(Q.T).dot(y)
        else:
            U, Sigma, V_T = np.linalg.svd(X)
            self.w = V_T.T.dot(np.linalg.inv(Sigma)).dot(U.T).dot(y)
        return self.w

    def _compute_cost(self, X, y, w, m):
        cost = (1 / m) * np.sum(np.square(y - X.dot(w)))
        return cost

    def _gradient_descent(self, X, y, alpha, num_iters, epsilon):
        m = len(y)
        self.w = np.zeros((X.shape[1], 1))  # Инициализируем веса
        cost_history = [0]

        for i in range(num_iters):
            y_hat = X.dot(self.w)
            errors = y - y_hat
            gradient = - (2 / m) * X.T.dot(errors)
            self.w -= alpha * gradient

            # На каждом шагу будем записывать функцию потерь
            cost = self._compute_cost(X, y, self.w, m)
            cost_history.append(cost)

            if np.abs(cost_history[-1] - cost_history[-2]) < epsilon:
                return self.w

            # Распечатать стоимость
            if i % 25 == 0:
                print(f"Iteration {i}: Cost {cost}")
        print(self.w)
        return self.w

    def fit(self, X_train, y_train, method='gradient_descent', alpha=0.0002, num_iters=1000, epsilon=1e-7):
        # Преобразуем данные в numpy массивы, если они являются DataFrame или Series
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.to_numpy().reshape(-1, 1)
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("Входные данные содержат недопустимые значения (NaN)")
        if method == 'gradient_descent':
            return self._gradient_descent(X_train, y_train, alpha, num_iters, epsilon)
        elif method == 'normal_eq':
            return self._normal_equation(X_train, y_train)
        else:
            raise ValueError('Не выбран метод расчета')

    def predict(self, X_test):
        # Преобразуем данные в numpy массивы, если они являются DataFrame или Series
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            X_test = X_test.to_numpy()
        if np.any(np.isnan(X_test)):
            raise ValueError("Входные данные содержат недопустимые значения (NaN)")
        if self.w is not None:
            return X_test.dot(self.w)
        raise ValueError("Сначала модель надо обучить")
