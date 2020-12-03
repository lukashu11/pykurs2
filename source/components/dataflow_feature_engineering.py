import pandas as pd
from functools import reduce

def calc_count(df, groupkey, dict):
    df = df.groupby(groupkey).count().reset_index()
    df = df.rename(columns=dict)
    return df


def calc_mean(df, groupkey, dict):
    df = df.groupby(groupkey).mean().reset_index()
    df = df.rename(columns=dict)
    return df


def join_dfs(df_left, df_right, join_key, drop_nan):
    df = pd.merge(df_left, df_right, on=join_key, how='left')
    df = df.dropna(subset=drop_nan)
    return df


def merge_calc_cols(df_to_merge, join_key):
    df = reduce(lambda left, right: pd.merge(left, right, on=join_key, how='left'), df_to_merge)
    df = df.dropna()
    return df

