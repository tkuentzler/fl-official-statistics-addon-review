"""umlaut_lte.py"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, ShuffleSplit
from tensorflow import keras
from keras import layers
from tensorflow_addons.metrics import RSquare
import keras_tuner
import matplotlib.pyplot as plt

def write_result(result: str, result_file="./results.txt"):
    """Prints results to terminal and writes to file.
    
    :param result: The data to store
    :type result: str
    :param result_file: The file to store the result in
    :type result_file: str, optional
    """
    print(result)
    with open(result_file, 'a') as f:
        f.write(result)
        f.write('\n')


def get_data_csv(input_data_file="data/lte.csv", threshold_min_days_per_user=0,
                 test_split_size=0.2, threshold_min_datpoints_per_day=10):
    """Import data and prepare for further processing.

    :param input_data_file: The path to the input data parquet file
    :type input_data_file: str, optional
    :param threshold_min_days_per_user: How many days a user should be present
        in the database to be used for the model
    :type threshold_min_days_per_user: int, optional
    :param test_split_size: The fraction how much data should be used for testing
    :type test_split_size: float, optional
    :param threshold_min_datpoints_per_day: The minimum amount of datapoints a user
        should contain to be used for model training
    :type threshold_min_datpoints_per_day: int, optional
    :return: lists containing the train/test splits: X_train, X_test, y_train, y_test
    :rtype: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
    """
    # DATA IMPORT
    df = pd.read_csv(input_data_file)

    # DATA PREPROCESSING

    write_result(f'running with threshold {threshold_min_days_per_user},'
                 f'data point threshold {threshold_min_datpoints_per_day},'
                 f'and test split {test_split_size}...')

    # choose one of the two target variables:
    target_variable = 'radius_activity_user'
    # target_variable = 'radius_activity_state'

    write_result(f'target variable: {target_variable}')

    # ALL features in the data set:
    all_features = df.columns.to_list()

    # features NOT used as input features for the model
    not_training_features = [
        'id',
        'date',
        # possible target features:
        'radius_activity_user',
        'radius_activity_state',
        # ignore this feature for now:
        'velocity_mean'
    ]

    write_result(f'Ignoring velocity? {"velocity_mean" in not_training_features}')

    training_features = [e for e in all_features if e not in not_training_features]

    # features which will be scaled from 0 to 1 for training
    scale_features = [
        'calendar_week',
        'weekday',
        'month',
        'count_data_points',
        'rsrp_variance',
        'rsrq_variance',
        'rssnr_variance',
        'rssi_variance',
        'rsrp_mean',
        'rsrq_mean',
        'rssnr_mean',
        'rssi_mean',
        'velocity_mean'
    ]

    write_result(f'before filtering data: {len(df.index)} rows in data set.')

    # remove data from users with too few data points:
    df = df[(df['count_data_points'] > threshold_min_datpoints_per_day)]
    count_users = df['id'].value_counts()
    users_below_count_threshold = []
    for id, value in count_users.items():
        if value < threshold_min_days_per_user:
            users_below_count_threshold.append(id)
    df = df.drop(df[df['id'].isin(users_below_count_threshold)].index)

    write_result(f'after filtering data: {len(df.index)} rows in data set.')

    # scale features to [0,1]:
    scaler = RobustScaler()
    df[scale_features] = scaler.fit_transform(df[scale_features])

    # split data into train and test set
    # at random for now -- if there is a more useful way, we should change this
    return train_test_split(df[training_features], df[[target_variable]], test_size=test_split_size, shuffle=True, random_state=0)


def get_data_parquet(input_data_file="data/df_fl_excerpt.parquet", threshold_min_days_per_user=0,
             test_split_size=0.2, threshold_min_datpoints_per_day=10):
    """Import data and prepare for further processing.

    :param input_data_file: The path to the input data parquet file
    :type input_data_file: str, optional
    :param threshold_min_days_per_user: How many days a user should be present
        in the database to be used for the model
    :type threshold_min_days_per_user: int, optional
    :param test_split_size: The fraction how much data should be used for testing
    :type test_split_size: float, optional
    :param threshold_min_datpoints_per_day: The minimum amount of datapoints a user
        should contain to be used for model training
    :type threshold_min_datpoints_per_day: int, optional
    :return: lists containing the train/test splits: X_train, X_test, y_train, y_test
    :rtype: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
    """
    # DATA IMPORT
    df = pd.read_parquet(input_data_file)

    # DATA PREPROCESSING

    # Added from Marten, should be checked if only floats have NaN
    df.fillna(0.0, inplace=True)

    # df.drop(columns=["Calendar_week", "Month"], inplace=True)

    write_result(f'running with threshold {threshold_min_days_per_user},'
                 f'data point threshold {threshold_min_datpoints_per_day},'
                 f'and test split {test_split_size}...')

    # choose one of the two target variables:
    target_variable = 'Radius_activity_user'
    # target_variable = 'Radius_activity_state'

    write_result(f'target variable: {target_variable}')

    # ALL features in the data set:
    all_features = df.columns.to_list()

    # features NOT used as input features for the model
    not_training_features = [
        'Id',
        'Date',
        # possible target features:
        'Radius_activity_user',
        'Radius_activity_state',
        # ignore this feature for now:
        'Velocity_mean'
    ]

    write_result(f'Ignoring velocity? {"Velocity_mean" in not_training_features}')

    training_features = [e for e in all_features if e not in not_training_features]

    # features which will be scaled from 0 to 1 for training
    scale_features = [
        'Calendar_week',
        'Weekday',
        'Month',
        'Count_data_points',
        'Rsrp_variance',
        'Rsrq_variance',
        'Rssnr_variance',
        'Rssi_variance',
        'Rsrp_mean',
        'Rsrq_mean',
        'Rssnr_mean',
        'Rssi_mean',
        'Velocity_mean'
    ]


    write_result(f'before filtering data: {len(df.index)} rows in data set.')
    print(df.keys())
    # remove data from users with too few data points:
    df = df[(df['Count_data_points'] > threshold_min_datpoints_per_day)]
    count_users = df['Id'].value_counts()
    users_below_count_threshold = []
    for id, value in count_users.items():
        if value < threshold_min_days_per_user:
            users_below_count_threshold.append(id)
    df = df.drop(df[df['Id'].isin(users_below_count_threshold)].index)

    write_result(f'after filtering data: {len(df.index)} rows in data set.')

    # scale features to [0,1]:
    scaler = RobustScaler()
    df[scale_features] = scaler.fit_transform(df[scale_features])

    # split data into train and test set
    # at random for now -- if there is a more useful way, we should change this
    return train_test_split(df[training_features], df[[target_variable]], test_size=test_split_size, shuffle=True, random_state=0)


def plot_loss(history, y_min=None, y_max=None, loss_fig_path="./loss_fig.png"):
    """Plots the given model history.

    :param history: The training history of the model
    :type history: keras.callbacks.History
    :param y_min: The y scale lower limit
    :type y_min: float, optional
    :param y_max: The y scale upper limit
    :type y_max: float, optional
    """
    plt.plot(history.history['r_square'], label='train R^2')
    plt.plot(history.history['val_r_square'], label='test R^2')
    if y_min is not None and y_max is not None:
        plt.ylim([y_min, y_max])
    plt.xlabel('Epoch')
    plt.ylabel('r_square')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_fig_path)