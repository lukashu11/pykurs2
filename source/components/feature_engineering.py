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

def split_data(df):

    ## Get X and y
    X = df.drop(columns=['is_canceled', 'order_hour_minute_second', 'order_year_month_day']).values
    y = df['is_canceled'].values

    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)

    # Feature Scaling (normalizing the data)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Feature Engineering -  PCA for dimension reduction (Real coordinate space: metric/ binary)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test, y_train, y_test, explained_variance
