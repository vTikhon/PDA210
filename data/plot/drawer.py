import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer.stat_criterion import StatCriteria


class Drawer:

    """
    Автоматически поворачивает метки оси X, если они накладываются.
    """
    @staticmethod
    def auto_rotate_xticks():
        labels = plt.gca().get_xticklabels()  # Получаем текущие метки на оси X
        label_lengths = [len(label.get_text()) for label in labels]  # Считаем длину текста в метках
        max_length = max(label_lengths)  # Максимальная длина текста
        if 10 < max_length <= 15:  # Если длина текста слишком большая, выбираем угол поворота
            rotation = 45
            plt.xticks(rotation=rotation)
        elif max_length > 15:
            rotation = 90
            plt.xticks(rotation=rotation)
        else:
            plt.xticks(rotation=0)

    """
    SEABORN
    Строит KDE, гистограмму, и boxplot для всех числовых столбцов в DataFrame.
    """
    @staticmethod
    def plot_numeric_seaborn(df):
        for column in df.select_dtypes(include='number').columns:
            plt.figure(figsize=(16, 4))
            sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})

            plt.subplot(1, 3, 1)
            sns.kdeplot(data=df, x=column, color='green')

            plt.subplot(1, 3, 2)
            sns.histplot(data=df, x=column, color='blue')

            plt.subplot(1, 3, 3)
            sns.boxplot(data=df, x=column, color='red')

            plt.show()

    """
    SEABORN
    Строит barplot и countplot для всех категориальных столбцов в DataFrame.
    """
    @staticmethod
    def plot_categorical_seaborn(df, target_col=None):
        for column in df.select_dtypes(include='object').columns:
            plt.figure(figsize=(18, 5))
            sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})

            # Countplot
            plt.subplot(1, 2, 1)
            sns.countplot(data=df, x=column, color='skyblue', hue=target_col)
            plt.title(f"Countplot - {column}")
            Drawer.auto_rotate_xticks()

            # Barplot (если есть целевая переменная для сравнения)
            if target_col and target_col in df.columns:
                plt.subplot(1, 2, 2)
                sns.barplot(data=df, x=column, y=target_col, palette='Set2')
                plt.title(f"Barplot - {column} vs {target_col}")
                Drawer.auto_rotate_xticks()
            else:
                plt.subplot(1, 2, 2)
                sns.countplot(data=df, x=column, color='lightcoral')
                plt.title(f"Countplot - {column} (no target)")
                Drawer.auto_rotate_xticks()

            plt.tight_layout()
            plt.show()

    """
    PLOTLY
    Строит гистограмму и boxplot для всех числовых столбцов в DataFrame.
    """
    @staticmethod
    def plot_numeric_plotly(df):
        for col in df.select_dtypes(include='number').columns:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Boxplot"])

            fig.add_trace(
                go.Histogram(x=df[col], name="Histogram"),
                row=1, col=1
            )

            fig.add_trace(
                go.Box(x=df[col], name="Boxplot"),
                row=1, col=2
            )

            fig.update_layout(height=500, width=1400, title_text=f"Distribution of {col}")
            fig.show()

    """
    PLOTLY
    Строит гистограммы для всех категориальных столбцов в DataFrame.
    """
    @staticmethod
    def plot_categorical_plotly(df, target_col=None):
        for col in df.select_dtypes(include='object').columns:
            fig = px.histogram(
                df,
                x=col,
                color=target_col if target_col in df.columns else None,
                barmode="group",
                height=500,
                width=1000,
                title=f"Distribution of {col}"
            )
            fig.show()

    @staticmethod
    def plot_heatmap(df, method='spearman', boundary = 0.5):
        corr = df.corr(method=method, numeric_only=True)
        mask = (corr < boundary) & (corr > -boundary)  # Маска для скрытия слабых корреляций

        plt.figure(figsize=(13, 5))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            vmin=-1,
            vmax=1,
            cmap='coolwarm',
            mask=mask  # Применяем маску
        )
        plt.show()

    @staticmethod
    def plot_heatmap_category(df):
        categorical_features = df.select_dtypes(include='object').columns
        corr_matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)
        for i in categorical_features:
            for j in categorical_features:
                if i == j:
                    corr_matrix.loc[i, j] = 0
                else:
                    groups = [df[i], df[j]]
                    stat, p_value, _, _ = StatCriteria().chi2_contingency(groups)
                    corr_matrix.loc[i, j] = stat if stat is not None else 0

        corr_matrix = corr_matrix.astype(float)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, vmin=0)
        plt.title('Тепловая карта корреляции категориальных признаков (stat chi2_contingency)')
        plt.show()

