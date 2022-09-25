import pandas as pd

from const import TIMES


def date_to_week(data_value: pd.Series) -> pd.Series:
    return data_value.weekday()


def month(data_value: pd.Series) -> pd.Series:
    return data_value.month


def hour(data_value: pd.Series) -> pd.Series:
    return data_value.hour


def minute(data_value: pd.Series) -> pd.Series:
    return data_value.minute


def preprocess_time(df: pd.DataFrame) -> pd.DataFrame:  # параметры для отсечения времени
    df['day_of_the_week'] = df['time1'].apply(date_to_week) + 1  # series or value from series?
    df['weekend'] = df['day_of_the_week'] > 5
    df['weekend'] = df['weekend'].astype(int)

    df['num_of_month'] = df['time1'].apply(month)
    df['hour'] = df['time1'].apply(hour)

    df['first_time'] = df[TIMES].min(axis=1)
    df['last_time'] = df[TIMES].max(axis=1)
    df['session_time'] = (df['last_time'] - df['first_time']).dt.seconds

    df['morning'] = df['hour'] < 12
    df['day'] = (df['hour'] >= 12) & (df['hour'] < 16)
    df['evening'] = (df['hour'] >= 16) & (df['hour'] < 19)
    df['night'] = df['hour'] >= 19
    df[['morning', 'day', 'evening', 'night']] = df[['morning', 'day', 'evening', 'night']].astype(int)
    return df

