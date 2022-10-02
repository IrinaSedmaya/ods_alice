from typing import Tuple, Set

import pandas as pd

from const import SITES


def unique_sites_search(df: pd.DataFrame) -> Tuple[Set[int], Set[int]]:
    """
    Выявляет сайты, которые посещает только Алиса или остальные пользователи

    :param df: признаки обучающей выборки
    :return: два множества уникальных сайтов
    """
    unique_sites_alice = set()
    unique_sites_other = set()
    for i in SITES:  # использовать константу
        unique_sites_alice |= set(df[df['target'] == 1][i])
    for i in SITES:
        unique_sites_other |= set(df[df['target'] == 0][i])

    unique_sites_alice, unique_sites_other = (
        unique_sites_alice - unique_sites_other, unique_sites_other - unique_sites_alice
    )
    return unique_sites_alice, unique_sites_other


def unique_sites_insert(df: pd.DataFrame, unique_sites_alice: Set[int], unique_sites_other: Set[int]) -> pd.DataFrame:
    """
    Создаёт в датасете два признака - уникальные сайты Алисы, уникальные сайты остальных

    :param df: признаки обучающей выборки
    :param unique_sites_alice: множество уникальных сайтов Алисы
    :param unique_sites_other: множество уникальных сайтов остальных
    :return: датафрейм с двумя новыми признаками
    """
    df['unique_sites_alice'] = 0
    df['unique_sites_other'] = 0
    df.loc[df[SITES].isin(unique_sites_alice).any(axis=1), 'unique_sites_alice'] = 1  # any
    df.loc[df[SITES].isin(unique_sites_other).any(axis=1), 'unique_sites_other'] = 1  # any
    # np.where
    return df
