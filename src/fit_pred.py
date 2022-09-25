import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


def fit_predict(x_train: csr_matrix, y_train: pd.Series, x_test: csr_matrix) -> pd.Series:
    lr = LogisticRegression(random_state=17, max_iter=500, C=0.71, n_jobs=-1).fit(x_train, y_train)
    return lr.predict_proba(x_test)[:, 1]

