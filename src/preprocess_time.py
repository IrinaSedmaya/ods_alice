import pandas as pd

from const import TIMES


def date_to_week(data_value: pd.Series) -> pd.Series:
    """
    Определяет день недели

    :param data_value: дата в формате день/месяц/год
    :return: день недели
    """
    return data_value.weekday()


def month(data_value: pd.Series) -> pd.Series:
    """
    Определяет месяц

    :param data_value: дата в формате день/месяц/год
    :return: месяц
    """
    return data_value.month


def hour(data_value: pd.Series) -> pd.Series:
    """
    Определяет который час

    :param data_value: время в формате час/минута/секунда
    :return: час
    """
    return data_value.hour


def minute(data_value: pd.Series) -> pd.Series:
    """
    Определяет минуты

    :param data_value: время в формате час/минута/секунда
    :return: минуты
    """
    return data_value.minute


def preprocess_time(df: pd.DataFrame, friday=5, midday=12, evening_time=16, night_time=19) -> pd.DataFrame:
    """
    Формирует новые признаки

    :param df: признаки обучающей выборки
    :param friday: пятый день недели, отсекающий выходные
    :param midday: отсекает утренние часы
    :param evening_time: отсекает верние часы
    :param night_time: отсекает ночные часы
    :return: датафрейм с одиннадцатью новыми признаками
    """
    df['day_of_the_week'] = df['time1'].apply(date_to_week) + 1  # series or value from series?
    df['weekend'] = df['day_of_the_week'] > friday
    df['weekend'] = df['weekend'].astype(int)

    df['num_of_month'] = df['time1'].apply(month)
    df['hour'] = df['time1'].apply(hour)

    df['first_time'] = df[TIMES].min(axis=1)
    df['last_time'] = df[TIMES].max(axis=1)
    df['session_time'] = (df['last_time'] - df['first_time']).dt.seconds

    df['morning'] = df['hour'] < midday
    df['day'] = (df['hour'] >= midday) & (df['hour'] < evening_time)
    df['evening'] = (df['hour'] >= evening_time) & (df['hour'] < night_time)
    df['night'] = df['hour'] >= night_time
    df[['morning', 'day', 'evening', 'night']] = df[['morning', 'day', 'evening', 'night']].astype(int)
    return df
