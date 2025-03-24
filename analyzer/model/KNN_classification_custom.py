import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances


class KNNClassificationCustom:

    """
    :param k: Количество соседей.
    :param metric: Метрика расстояния ('euclidean', 'manhattan', 'cosine' и т.д.).
    :param weights: Веса соседей ('uniform' или 'distance').
    """
    def __init__(self, n_neighbors=3, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    """
    Сохраняет обучающие данные.
    :param X_train: Обучающие данные (признаки).
    :param y_train: Обучающие данные (метки классов).
    """
    def fit(self, X_train, y_train):
        self.X_train = self._convert_pandas_to_numpy(X_train)
        self.y_train = self._convert_pandas_to_numpy(y_train).ravel()

    """
    Предсказывает класс для каждого объекта в X_test.
    :param X_test: Тестовые данные (признаки).
    :return: Предсказанные метки классов.
    """
    def predict(self, X_test):
        X_test = self._convert_pandas_to_numpy(X_test)
        return np.array([self._predict(x) for x in X_test])

    """
    Предсказывает класс для одного объекта.
    :param x: Объект для классификации.
    :return: Предсказанный класс.
    """
    def _predict(self, x):
        # Вычисляем расстояния между x и всеми объектами в X_train
        distances = pairwise_distances([x], self.X_train, metric=self.metric)[0]

        # Находим индексы k ближайших соседей
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Взвешенное голосование (если weights='distance')
        if self.weights == 'distance':
            k_distances = distances[k_indices]
            weights = 1 / (k_distances + 1e-10)  # Добавляем небольшое значение для избежания деления на ноль
            weighted_votes = Counter(dict(zip(k_nearest_labels, weights)))
            most_common = weighted_votes.most_common(1)
        else:
            # Обычное голосование (если weights='uniform')
            most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]

    """
    Преобразует данные в массив NumPy, если они в формате DataFrame.
    :param data: Данные (признаки или метки).
    :return: Данные в формате массива NumPy.
    """
    def _convert_pandas_to_numpy(self, data):
        if isinstance(data, pd.DataFrame):
            # Если данные в формате DataFrame, преобразуем их в массив NumPy
            return data.values
        return data