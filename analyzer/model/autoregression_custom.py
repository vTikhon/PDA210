import numpy as np


class AutoregressionCustom:
    '''Класс самописной АР.
    Parametres:
    series: временной ряд
    lag_order: количество членов авторегрессии
    '''

    def __init__(self, series, lag_order):
        self.series = series
        self.lag_order = lag_order
        self.coef = []

    def fit(self):
        '''Метод расчета матрица признаков и обучения модели.'''
        n = len(self.series)
        F = np.array([self.series[i:i + self.lag_order] for i in range(n - self.lag_order)])
        y = self.series[self.lag_order:]
        self.coef = np.array(np.linalg.lstsq(F, y)[0])

    def forecast(self, steps):
        '''Предсказание АР
        Return:
        last_values: предсказанные значения
        '''
        init_calculate = np.array(self.series[-self.lag_order:])
        last_values = []
        for _ in range(steps):
            next_value = self.coef.dot(init_calculate.T)
            last_values.append(next_value)
            init_calculate = np.roll(init_calculate, -1)
            init_calculate[-1] = next_value
        return np.array(last_values)
