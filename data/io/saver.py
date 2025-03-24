import os

class Saver:

    @staticmethod
    def save_csv(df, filename, index=False):
        # Проверяем, существует ли директория, если нет - создаем
        dir_path = './dataset/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Формируем полный путь к файлу
        file_path = os.path.join(dir_path, f"{filename}.csv")
        # Сохраняем DataFrame в CSV
        df.to_csv(file_path, index=index)
        print(f"Файл успешно сохранён: {file_path}")
