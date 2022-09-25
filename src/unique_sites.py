from typing import List, Tuple, Set

import pandas as pd

from const import SITES


def unique_sites_search(df: pd.DataFrame) -> Tuple[Set[int], Set[int]]:
    unique_sites_alice = set()
    unique_sites_other = set()
    for i in range(1, 11):  # использовать константу
        unique_sites_alice |= set(df[df['target'] == 1][f'site{i}'])
    for i in range(1, 11):
        unique_sites_other |= set(df[df['target'] == 0][f'site{i}'])

    unique_sites_alice, unique_sites_other = (
        unique_sites_alice - unique_sites_other, unique_sites_other - unique_sites_alice
    )
    return unique_sites_alice, unique_sites_other


def unique_sites_insert(df: pd.DataFrame, unique_sites_alice: Set[int], unique_sites_other: Set[int]) -> pd.DataFrame:
    df['unique_sites_alice'] = 0
    df['unique_sites_other'] = 0
    df.loc[df[SITES].isin(unique_sites_alice).any(axis=1), 'unique_sites_alice'] = 1  # any
    df.loc[df[SITES].isin(unique_sites_other).any(axis=1), 'unique_sites_other'] = 1  # any
    # np.where
    return df

