from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, chisquare, norm, ttest_1samp, ttest_ind, shapiro
from scipy.stats import wilcoxon, friedmanchisquare, shapiro
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


class StatCriteria:

    def __init__(self, alpha=0.05):
        self.alpha = alpha


    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=1
    # Одновыборочный z-критерий (параметрический для непрерывных данных)
    # z-критерий используется для проверки того,
    # является ли среднее значение генеральной совокупности (N=1, то есть одной выборки)
    # меньше, больше или равно некоторому определенному значению.
    # Нулевая гипотеза (H0): Среднее значение выборки равно заданному значению.
    def z_criteria(self, df_feature, hypothesized_mean, std):
        # Рассчитаем выборочное среднее, размер выборки, и стандартную ошибку
        sample_mean = np.mean(df_feature)
        sample_size = len(df_feature)
        standard_error = std / np.sqrt(sample_size)
        # Z-статистика
        stat = (sample_mean - hypothesized_mean) / standard_error
        # p-value для двухстороннего теста
        p_value = 2 * norm.sf(abs(stat))
        print(f"Test z-statistic")
        print(f'z-statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Среднее значение выборки не равно заданному значению (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Среднее значение выборки равно заданному значению (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=1
    # Критерий Шапиро-Уилка соответствия нормальному распределению
    # Нулевая гипотеза (H0): Данные соответствуют нормальному распределению.
    def shapiro(self, df_feature):
        df_feature = df_feature.dropna()
        if len(df_feature) == 0:
            raise ValueError("Выборка пуста после исключения пропусков.")
        stat, p_value = shapiro(df_feature)
        print(f"Shapiro-Wilk Test")
        print(f'statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Данные не соответствуют нормальному распределению (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Данные соответствуют нормальному распределению (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=1
    # Критерий Стьюдента (одновыборочный)
    # Нулевая гипотеза (H0): Среднее значение выборки равно заданному значению.
    def ttest_1samp(self, df_feature, hypothesized_mean):
        df_feature = df_feature.dropna()
        if len(df_feature) == 0:
            raise ValueError("Выборка пуста после исключения пропусков.")
        stat, p_value = ttest_1samp(df_feature, hypothesized_mean)
        print(f"One Samples T-test")
        print(f't-statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Среднее значение выборки не равно заданному значению (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Среднее значение выборки равно заданному значению (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=2
    # Критерий Стьюдента (параметрический для непрерывных данных)
    # это статистический метод, который позволяет сравнивать средние значения двух выборок и на основе результатов теста делать
    # заключение о том, различаются ли они друг от друга статистически или нет.
    # Нулевая гипотеза (H0): Средние значения в двух выборках равны (отсутствуют статистически значимые различия).
    def ttest_ind(self, groups):
        if len(groups) != 2:
            raise ValueError("Для критерия Стьюдента требуется ровно две группы.")
        stat, p_value = ttest_ind(*groups)
        print("Independent Samples T-test")
        print(f't-statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Средние значения в двух выборках не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Средние значения в двух выборках равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=2
    # U-критерий Манна-Уитни используется для сравнения различий между ДВУМЯ (N=2) независимыми выборками,
    # когда распределение выборки не является нормальным, а размеры выборки малы
    # Нулевая гипотеза (H0): Распределения двух выборок равны (отсутствуют статистически значимые различия).
    def mannwhitneyu(self, groups, alternative='two-sided'):
        if len(groups) != 2:
            raise ValueError("Для теста Манна-Уитни требуется ровно две группы.")
        stat, p_value = mannwhitneyu(*groups, alternative=alternative)
        print('Mann-Whitney U test')
        print(f'U_statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Распределения двух выборок не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Распределения двух выборок равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N>=3
    # Критерий Краскела - Уоллиса (непараметрический для непрерывных данных)
    # Критерий Краскела - Уоллиса используемый для определения, есть ли статистически значимые различия
    # между медианами ТРЁХ И БОЛЕЕ (n>=3) независимых групп
    # Нулевая гипотеза (H0): Медианы всех групп равны (отсутствуют статистически значимые различия).
    def kruskal(self, groups):
        if len(groups) < 2:
            raise ValueError("Для теста Краскела-Уоллиса требуется хотя бы две группы.")
        stat, p_value = kruskal(*groups)
        print('kruskal')
        print(f'statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Медианы всех групп не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Медианы всех групп равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=1
    # Критерий хи-квадрат
    # Хи-квадрат критерия согласия — используется для определения того,
    # следует ли категориальная переменная (N=1 - то есть одна выборка) гипотетическому распределению.
    # Нулевая гипотеза (H0): Наблюдаемые частоты согласуются с ожидаемыми частотами (нет статистически значимых различий).
    def chisquare(self, df, feature_name, target_name):
        groups = self._prepare_groups(df, feature_name, target_name)
        if len(feature_name) != len(target_name):
            raise ValueError("Массивы наблюдаемых и ожидаемых частот должны быть одинаковой длины.")
        stat, p_value = chisquare(*groups)
        print("Chi-square test statistic")
        print(f'chisq_statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Наблюдаемые частоты не согласуются с ожидаемыми частотами (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Наблюдаемые частоты согласуются с ожидаемыми частотами (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий хи-квадрат
    # Критерий независимости хи-квадрат — используется для определения наличия значимой связи
    # между ДВУМЯ (N=2) категориальными переменными
    # Нулевая гипотеза (H0): Две переменные независимы (отсутствуют статистически значимые связи).
    def chi2_contingency(self, groups):
        if any(len(group.unique()) < 2 for group in groups):
            raise ValueError("Ошибка: должно быть минимум 2 уникальных значения в каждой переменной.")
        contingency_table = pd.crosstab(*groups)
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        print('Chi-square test')
        print(f'chisq_statistic = {stat:.3f}, dof = {dof:.3f}')
        if p_value < self.alpha:
            print(f"Две переменные зависимы (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Две переменные независимы (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value, dof, expected

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ИЛИ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий Вилкоксона (непараметрический для непрерывных и порядковых данных)
    # Критерий Вилкоксона используется для сравнения ДВУХ (N=2) связанных (зависимых) выборок
    # по количественному или порядковому признаку.
    # Нулевая гипотеза (H0): Различия между парами значений равны (отсутствуют статистически значимые различия).
    def wilcoxon(self, groups):
        if len(groups) != 2:
            raise ValueError("Для теста Вилкоксона требуется ровно две группы.")
        stat, p_value = wilcoxon(*groups)
        print('Wilcoxon test')
        print(f'Wilcoxon-statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Различия между парами значений не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Различия между парами значений равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ИЛИ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N>=3
    # Критерий Фридмана (непараметрический для непрерывных и порядковых данных)
    # Критерий Фридмана используется для сравнения ТРЁХ И БОЛЕЕ (N>=3) связанных (зависимых) выборок
    # по количественному или порядковому признаку.
    # Нулевая гипотеза (H0): Распределения во всех группах равны (отсутствуют статистически значимые различия).
    def friedmanchisquare(self, groups):
        if len(groups) < 2:
            raise ValueError("Для теста Фридмана требуется хотя бы две группы.")
        stat, p_value = friedmanchisquare(*groups)
        print('Friedman test')
        print('statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Распределения во всех группах не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Распределения во всех группах равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий Мак-Нимара (для категориальных зависимых данных)
    # Тест Макнемара используется для определения наличия статистически значимой разницы в пропорциях
    # между ПАРНЫМИ (N=2) данными.
    # Нулевая гипотеза (H0): Доли согласия в двух связанных группах равны (отсутствуют статистически значимые различия).
    def mcnemar(self, groups):
        if any(len(group.unique()) < 2 for group in groups):
            raise ValueError("Ошибка: должно быть минимум 2 уникальных значения в каждой переменной.")
        contingency_table = pd.crosstab(*groups)
        print("McNemar's test")
        result = mcnemar(contingency_table, exact=True)  # Используется exact=True для точного теста (если требуется)
        print(result)
        return result

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N>=3
    # Q-тест Кокрана (Cochran's Q test) — это статистический тест,
    # используемый для определения наличия различий в доли успеха в нескольких (N>=3) связанных группах.
    # Q-тест Кокрана применяется, когда данные являются бинарными (успех/неудача) и измерены повторно в различных условиях.
    def cochrans_q(self, groups):
        if any(len(group.unique()) < 2 for group in groups):
            raise ValueError("Ошибка: должно быть минимум 2 уникальных значения в каждой переменной.")
        contingency_table = pd.crosstab(*groups)
        print("Cochran's Q test")
        result = cochrans_q(contingency_table)
        print(result)
        return result

    # Допущение о мультиколлинеарности
    def VIF(self, df, target_name):
        numeric_col = df.drop(columns=target_name).describe().columns
        vif_data = pd.DataFrame()
        vif_data.index = numeric_col
        vif_data['VIF'] = [variance_inflation_factor(df[numeric_col].values, i) for i in range(len(numeric_col))]
        return vif_data

