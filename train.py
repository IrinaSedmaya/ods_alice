import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from const import DATA_DIR, SITES, TIMES
from src.fit_pred import fit_predict
from src.matrix import create_matrix
from src.preprocess_time import preprocess_time
from src.unique_sites import unique_sites_search, unique_sites_insert
from src.utils import cyclical, normalize

# ко всему - аннотации типов, докстринги, парсинг аргументов, форматтер, ридми, гитигнор, скрипт для загрузки


# вынести в функцию, смысловые блоки

train = pd.read_csv(DATA_DIR / 'train.csv', index_col='session_id', parse_dates=TIMES).sort_values(by='time1')
train_target = train['target']

train[SITES] = train[SITES].fillna(0).astype(int)
uniq_sites_alice, uniq_sites_other = unique_sites_search(train)
train = unique_sites_insert(train, uniq_sites_alice, uniq_sites_other)

train = preprocess_time(train)
train = pd.concat([train, cyclical(train[['num_of_month', 'hour', 'day_of_the_week']])], axis=1)
train['session_time'] = normalize(train['session_time'])

train = train.drop(columns=['day_of_the_week', 'num_of_month', 'hour', 'first_time', 'last_time', 'target'], axis=1)
train = train.fillna(0).drop(columns=TIMES, axis=1)

print(train.columns)

# columns order
train_times, train_days = train.loc[:, 'session_time': 'day_of_the_week_cos'], \
                          train.loc[:, 'unique_sites_alice': 'weekend']

all_train = create_matrix(train[SITES], train_days, train_times)

x_train, x_test, y_train, y_test = train_test_split(all_train, train_target, train_size=0.9, random_state=1)
# отделять раньше - отдельным скриптом

y_pred = fit_predict(x_train, y_train, x_test)

print(roc_auc_score(y_test, y_pred))

