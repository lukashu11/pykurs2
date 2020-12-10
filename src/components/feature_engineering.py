import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler


def calc_count(df_count_calc, groupkey, rename_dict):
    df_count_calc = df_count_calc.groupby(groupkey).count()
    df_count_calc = df_count_calc.rename(columns=rename_dict)
    return df_count_calc


def calc_mean(df_mean_calc, groupkey, rename_dict):
    df_mean_calc = df_mean_calc.groupby(groupkey).mean()
    df_mean_calc = df_mean_calc.rename(columns=rename_dict)
    return df_mean_calc


def join_dfs(df_left, df_right, join_key, drop_nan):
    df_joined = pd.merge(df_left, df_right, on=join_key, how='left')
    df_joined = df_joined.dropna(subset=drop_nan)
    return df_joined


def merge_calc_cols(df_to_merge, join_key):
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=join_key, how='left'), df_to_merge)
    df_merged = df_merged.dropna()
    return df_merged


def split_data(df_for_split):
    # Get X and y
    X = df_for_split.drop(columns=['order_status_canceled']).values
    y = df_for_split['order_status_canceled'].values

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    # Feature Scaling (normalizing the data)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def reduce_dimensions(X_train, X_test):
    # Feature Engineering -  PCA for dimension reduction (Real coordinate space: metric/ binary)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test, explained_variance


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


def split_new_data(new_data_df):
    # Get 5 samples of each classification option
    new_data_df = new_data_df.head(500)
    # Get X as numpy array
    new_data_array = new_data_df.drop(columns=['order_status_canceled']).values
    return new_data_array, new_data_df


def scale_features_new_data(new_data):
    # Feature Scaling (normalizing the data)
    sc = StandardScaler()
    new_data = sc.fit_transform(new_data)
    return new_data


def reduce_dimensions_new_data(new_data):
    # Feature Engineering -  PCA for dimension reduction (Real coordinate space: metric/ binary)
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(new_data)
    explained_variance = pca.explained_variance_ratio_
    return new_data, explained_variance
