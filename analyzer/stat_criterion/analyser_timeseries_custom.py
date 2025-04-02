import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots


class AnalyserTimeSeriesCustom:
    '''Анализ временного ряда.'''

    def __init__(self, ts):
        self.ts = ts

    def check_stat(self):
        '''Проверка стационарности по стат.критериям.'''
        p_adfuller = sm.tsa.stattools.adfuller(self.ts)[
            1]  # Тест Дики-Фуллура (нулевая гипотеза - временной ряд является нестационарным)
        p_kpss = sm.tsa.stattools.kpss(self.ts)[
            1]  # Тест Квятковского-Филлипса-Шмидта-Шина (нулевая гипотеза - временной ряд является стационарным)
        if p_adfuller < 0.05 and p_kpss > 0.05:
            print("Ряд стационарный")
            return True
        print("Ряд не стационарный")
        return False

    def decompositon(self, model, period):
        '''Декомпозиция ВР.

        :param model: тип модели
        :param period: период
        :return: None
        '''
        decompose = seasonal_decompose(self.ts, model=model, period=period)
        plt.figure(figsize=(15, 15))
        plt.subplot(411)
        plt.plot(self.ts, label='Оригинальный график')
        plt.grid()
        plt.legend()
        plt.subplot(412)
        plt.plot(decompose.trend, label='Тренд')
        plt.grid()
        plt.legend()
        plt.subplot(413)
        plt.plot(decompose.seasonal, label='Сезонность')
        plt.grid()
        plt.legend()
        plt.subplot(414)
        plt.plot(decompose.resid, label='Шум')
        plt.grid()
        plt.legend()

    def auto_correlation(self, lags):
        '''Построение АКФ и ЧАКФ.'''

        fig = tsaplots.plot_acf(self.ts, lags=lags)
        fig = tsaplots.plot_pacf(self.ts, lags=lags)

    def convert_to_stationary(self):
        '''Преобразование в стационарный вид.'''
        info = self.check_stat()
        if info:
            print('Преобразование не требуется')
        else:
            self.ts = self.ts.diff().dropna()
            self.convert_to_stationary(self)
        return self.ts
