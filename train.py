import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from const import DATA_DIR, SITES, TIMES, RANDOM_STATE
from src.fit_predict import fit
from src.matrix import create_matrix
from src.preprocess_time import preprocess_time
from src.unique_sites import unique_sites_search, unique_sites_insert
from src.utils import cyclical, normalize


def train(train, train_target, params: dict):
    train[SITES] = train[SITES].fillna(0).astype(int)
    uniq_sites_alice, uniq_sites_other = unique_sites_search(train)
    train = unique_sites_insert(train, uniq_sites_alice, uniq_sites_other)

    train = preprocess_time(train)
    train = pd.concat([train, cyclical(train[['num_of_month', 'hour', 'day_of_the_week']])], axis=1)
    train['session_time'] = normalize(train['session_time'])

    train = train.drop(columns=['day_of_the_week', 'num_of_month', 'hour', 'first_time', 'last_time', 'target'], axis=1)
    train = train.fillna(0).drop(columns=TIMES, axis=1)

    train_times, train_days = train[['session_time', 'morning', 'day', 'evening', 'night',
                                     'num_of_month_sin', 'num_of_month_cos', 'hour_sin', 'hour_cos',
                                     'day_of_the_week_sin', 'day_of_the_week_cos']], \
                              train[['unique_sites_alice', 'unique_sites_other', 'weekend']]

    all_train = create_matrix(train[SITES], train_days, train_times)

    # todo: dataclass
    return fit(x_train=all_train, y_train=train_target, params=params), unique_sites_alice, unique_sites_other

lr_model = train()