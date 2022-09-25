from typing import List

import pandas as pd
import scipy
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)


def to_csr_matrix(df: pd.DataFrame) -> csr_matrix:
    return csr_matrix(df.values)


def concat_matrix(matrix: List[csr_matrix]) -> csr_matrix:
    return hstack(matrix).tocsr()


def count_vectorizer(df: pd.DataFrame) -> csr_matrix:
    # noinspection PyTypeChecker
    df.to_csv('train_sites.txt', sep=' ', index=None, header=None)

    with open('train_sites.txt') as inp_train_file:
        train_sites = cv.fit_transform(inp_train_file)

    return train_sites


def create_matrix(sites: pd.DataFrame, days: pd.DataFrame, times: pd.DataFrame) -> csr_matrix:
    sites_matrix = count_vectorizer(sites)
    matrix = [sites_matrix, to_csr_matrix(days), to_csr_matrix(times)]
    all_train = concat_matrix(matrix)
    return all_train

