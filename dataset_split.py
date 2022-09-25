import pandas as pd
from sklearn.model_selection import train_test_split


def dataset_split(dataset_path, train_path, test_path):
    dataset = pd.read_csv(dataset_path, index_col='session_id')
    train, test = train_test_split(dataset, train_size=0.9, random_state=1)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

