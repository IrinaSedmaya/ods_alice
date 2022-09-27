import pandas as pd
from feature_engine.creation import CyclicalTransformer
cyclical_transformer = CyclicalTransformer(variables=None, drop_original=True)  # внести в функцию


def cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет cyclical_transformer к датафрейму

    :param df: признаки обучающей выборки
    :return: датафрейм
    """
    return cyclical_transformer.fit_transform(df)


def normalize(x: pd.Series) -> pd.Series:
    """
    Нормализует заданный признак

    :param x: признак обучающей выборки
    :return: pd.Series
    """
    x = x - x.min()
    return x / x.max()
