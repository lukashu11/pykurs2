import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler


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
    # Get X and y
    X = df.drop(columns=['order_status_canceled']).values
    y = df['order_status_canceled'].values

    # train_test_split
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
    return X, y, X_train, X_test, y_train, y_test, explained_variance


def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def implement_oversampling(X, X_train, y_train):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X_train, y_train)
    print(X_ros.shape[0] - X.shape[0], 'new random picked points')
    return X_ros, y_ros
