import pandas as pd
from feature_engine.creation import CyclicalTransformer
cyclical_transformer = CyclicalTransformer(variables=None, drop_original=True)  # внести в функцию


def cyclical(df: pd.DataFrame) -> pd.DataFrame:
    return cyclical_transformer.fit_transform(df)


def normalize(x: pd.Series) -> pd.Series:
    x = x - x.min()
    return x / x.max()

