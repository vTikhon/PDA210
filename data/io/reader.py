import logging
import os
import sys
import pandas as pd
from sklearn.utils import shuffle

logging.basicConfig(
    filename='app.log',  # Имя файла для записи логов
    level=logging.INFO,   # Уровень логирования
    format='%(asctime)s - %(levelname)s - %(message)s'  # Формат логов
)


class Reader:

    @staticmethod
    def read_csv(link):
        # Чтение CSV
        df = pd.read_csv(link)
        df = shuffle(df).reset_index(drop=True)
        df.columns = (df.columns
                      .str.replace(r"[ \-.,]", "_", regex=True)
                      .str.lower())
        return df

    @staticmethod
    def get_csv_path(subprojectdirectory, name_csv):
        try:
            # Определяем, запущено ли приложение из .exe
            if getattr(sys, 'frozen', False):
                # Если да, используем путь к папке dist/dataset
                base_dir = os.path.join(os.path.dirname(sys.executable), "dataset")
            else:
                # Если нет, используем путь относительно текущего скрипта
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                base_dir = os.path.join(base_dir, subprojectdirectory)

            # Возвращаем полный путь к CSV-файлу
            csv_path = os.path.join(base_dir, name_csv)
            logging.info(f"PATH to CSV-file: {csv_path}")
            return csv_path
        except Exception as e:
            logging.error(f"Error at path to CSV-file: {e}")
            raise
