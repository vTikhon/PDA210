import tkinter as tk
from tkinter import messagebox, scrolledtext
from launcher import LaunchPredict


class MainFrame:
    def __init__(self, title):
        self.root = tk.Tk()
        self.root.title(title)
        self.features = ["tmax"]  # Список признаков для ввода
        self.entries = {}  # Словарь для хранения полей ввода
        self._init_widgets()

    def _init_widgets(self):
        """Инициализация виджетов интерфейса."""
        self._create_input_fields()
        self._create_button()
        self._create_result_area()

    def _create_input_fields(self):
        """Создание полей ввода для каждого признака."""
        for feature in self.features:
            label = tk.Label(self.root, text=f"{feature}:")
            label.pack(pady=5)
            entry = tk.Entry(self.root)
            entry.pack(pady=5)
            self.entries[feature] = entry

    def _create_button(self):
        """Создание кнопки для запуска предсказания."""
        button = tk.Button(self.root, text="Search solution", command=self.on_button_click)
        button.pack(pady=10)

    def _create_result_area(self):
        """Создание области для вывода результатов."""
        self.result_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=40, height=10, state='disabled'
        )
        self.result_area.pack(pady=10)

    def on_button_click(self):
        """Обработчик нажатия кнопки."""
        try:
            data = self._collect_input_data()
            result = self._launch_prediction(data)
            self._display_result(result)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"You got an error: {e}")

    def _collect_input_data(self):
        """Сбор данных из полей ввода."""
        data = {}
        for feature, entry in self.entries.items():
            value = entry.get()
            if not value:
                raise ValueError(f"There is no '{feature}' in the field!")
            data[feature] = [value]
        return data

    def _launch_prediction(self, data):
        """Запуск предсказания с использованием LaunchPredict."""
        return LaunchPredict().launch_predict(data)

    def _display_result(self, result):
        """Отображение результата в текстовом поле."""
        self.result_area.config(state='normal')  # Разрешаем редактирование
        self.result_area.delete(1.0, tk.END)  # Очищаем предыдущий результат
        self.result_area.insert(tk.END, result)  # Вставляем новый результат
        self.result_area.config(state='disabled')  # Запрещаем редактирование

    def run(self):
        """Запуск основного цикла приложения."""
        self.root.mainloop()
