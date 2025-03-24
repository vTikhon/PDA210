import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logging.basicConfig(
    filename='app.log',  # Имя файла для записи логов
    level=logging.INFO,   # Уровень логирования
    format='%(asctime)s - %(levelname)s - %(message)s'  # Формат логов
)


class DataPreparation:

    def __init__(self):
        pass

    @staticmethod
    def train_test_split(df, exclude_features, target):
        df = shuffle(df).reset_index(drop=True)
        # Разделяем данные на X, y
        X = df.drop(exclude_features, axis=1)
        y = df[target]
        # Разделим данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def create_dataframe(data):
        if not data:
            raise ValueError("Entry data is empty!")

        X_new = pd.DataFrame(data)
        if isinstance(X_new, pd.DataFrame):
            logging.info("DataFrame has been successfully created.")
            return X_new
        else:
            raise TypeError("Impossible create DataFrame on input data!")