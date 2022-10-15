import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from const import RANDOM_STATE

def fit(x_train: csr_matrix, y_train: pd.Series, params: dict) -> LogisticRegression:
    return LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, **params).fit(x_train, y_train)


def predict(model: LogisticRegression, x_test: csr_matrix) -> pd.Series:
    """
    Делает предсказание

    :param x_test: признаки тестовой выборки
    :return: вероятности класса 1 для тестовой выборки
    """
    return model.predict_proba(x_test)[:, 1]
