"""pm25_beijing.py"""

import os
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support


# Data loader
class DataHandler:
    """Loads, preprocesses and handles the beijing pollution data. It is created for the beijing
    pollution data structure but can be used for any other data as long it is structured similarly.

    :param source: The path to load the data from
    :type source: str
    :param station: Station(s) for which the data should be loaded. If none is given, all stations
        are loaded. Otherwise only the given stations will be loaded which leads to make some
        preprocessing steps faster, however the scaling of the data might introduce small
        errors because in this case it is not the global min/max used for scaling the data but
        the min/max of the given station(s).
    :type station: str or list of str, optional
    :param features_to_use: Features from the data which should be loaded. If none is given, all
        features are loaded. features_to_use defines only the features actually used for the
        model input for prediction (and not the feature which will be predicted from the model).
    :type features_to_use: str or list of str, optional
    :param col_to_predict: The columns (or features) of the data which will be predicted from the
        model.
    :type col_to_predict: str or list of str, optional
    """

    def __init__(self, source, station=None, features_to_use=None, col_to_predict=None):
        """Constructor method
        """
        if col_to_predict is None:
            col_to_predict = ["PM2.5"]
        elif type(col_to_predict) == str:
            col_to_predict = [col_to_predict]
        if station is not None:
            print("Warning: If the station(s) is defined, not all data is loaded."
                  "This leads to faster processing but might result in "
                  "problems with the minmax scaler since it orientates "
                  "itself at the highest and lowest value (which might "
                  "differ for different stations. You can use the full "
                  "data and use the specific station(s) after the "
                  "preprocessing step.")
            if type(station) != list:
                station = [station]
        self.col_to_predict = col_to_predict
        self.data, self.features, self.station = self.load_data(source, station,
                                                  features_to_use, col_to_predict)
        self.model_input = None

    @staticmethod
    def load_data(data_path, stations=None, features=None, col_to_predict=None):
        """Loads the data from the file system into a dictionary of pandas dataframes.

        :param data_path: Path to load the data from
        :type data_path: str
        :param stations: If given, only the data of the specified stations will be loaded.
        :type stations: str of list of str, optional
        :param features: If given, only the data of the specified features will be loaded.
        :type features: str or list of str, optional
        :return: A dictionary with each station as key and a DataFrame as value, the features
            which were loaded, the stations which were loaded
        :rtype: dict, list of str, list of str
        """
        db = {}
        if features is not None:
            if col_to_predict is not None:
                features += col_to_predict
        for file in os.listdir(data_path):
            # This works only in the given database and might be changed for other names!
            # Uses the word between the second and last (or third) underscore "_"
            s = file[file.index("_", file.index("_") + 1) + 1:file.rindex("_")]
            if stations is not None:
                if s not in stations:
                    continue
            db[s] = pd.read_csv(data_path + file, usecols=features)
        if features is None:
            features = list(next(iter(db.values())).columns)
        if col_to_predict is not None:
            for c in col_to_predict:
                features.remove(c)
        if stations is None:
            stations = list(db.keys())
        return db, features, stations

    def preprocess_data(self, minmax_features=None, onehot_features=None, ignore_wd=False):
        """Fits the data into an interval [0;1] using the min-max scaler for given categories and
        one hot encoding (categorical) respectively. If wind direction ("wd") is in self.features
        there are automatically created the classes wd_N, wd_E, wd_S, and wd_W from and wd is
        removed. This happens independent of any other features (neither minmax or onehot).

        :param minmax_features: The features which should be min-max scaled
        :type minmax_features: list of str, optional
        :param onehot_features: The features which should be min-max scaled
        :type onehot_features: list of str, optional
        :param ignore_wd: Either to create the classes wd_N, wd_E, wd_S, and wd_W from wd or not
        :type ignore_wd: bool, optional
        """
        if not minmax_features and not onehot_features and "wd" not in self.features:
            raise ValueError("At least one of minmax_features/onehot_features needs to hold data")
        # Sollte 0 mm Regen eine zusätzliche Kategorie bekommen (True/False -> Regen)?
        scaler = MinMaxScaler()

        # Everything expects numerical data which does not hold for wd, so it
        # needs to be replaced by the north, east, south, west columns
        if not ignore_wd and "wd" in self.features:
            print("Recognized wd (wind direction) as feature. Create columns "
                  "north, east, south and west automatically.")
            self.create_wd_classes()
            print("Warning: 'wd' is not part of the features "
                  "anymore. Instead each wind direction is separated "
                  "(north, east, south, west)")
            self.features = [f for f in self.features + ["wd_N", "wd_E", "wd_S", "wd_W"] if f != "wd"]

        # Fit the data first - must be done over the complete set to be usable in a central model.
        # wd needs to be sort out because the values are not float.
        minmax_features = [f for f in minmax_features if f != "wd"]
        if len(self.data) != 1:
            reference_data = self.unite_data(self.data)
        else:
            reference_data = self.data[self.station[0]]

        scaler.fit(reference_data[minmax_features])
        for key in self.data.keys():
            self.data[key][minmax_features] = scaler.transform(self.data[key][minmax_features])

            if onehot_features:
                raise Exception("onehot not yet implemented")
            # for feature in onehot_features:
            # occurring_values = self.data[key][onehot_features].unique()
            # self.data[key][onehot_features] =

    @staticmethod
    def unite_data(data, stations=None, features=None):
        """Unites the given data into one DataFrame. The data is otherwise stored as one DataFrame
        per Station.
        :param data: The data to unify
        :type data: dict of DataFrames
        :param stations: The data of which stations should be unified. If none is given, all
            stations from data are used
        :type stations: list of str, optional
        :param features: The features of the data which should be unified. If none is given, all
            features are used
        :type features: list of str, optional
        :return: A single DataFrame with the data
        :rtype: DataFrame
        """
        if stations is None:
            stations = data.keys()

        united_data = pd.DataFrame()
        for s in stations:
            if features is None:
                united_data = pd.concat([united_data, data[s]], ignore_index=True)
            else:
                united_data = pd.concat([united_data, data[s][features]], ignore_index=True)
        return united_data

    def interpolate(self, features=None):
        """Fills in the missing data (NaNs) as there might be some empty data points.

        :param features: Defines for which features to interpolate the data. If none is given,
            all features are interpolated
        :type features: list of str, optional
        """
        if features is None:
            features = self.features
        for location in self.data:
            self.data[location][features] = self.data[location][features].interpolate(
                method="linear")  # , order=2)

    def check_for_nan(self, single_station=True):
        """Checks for NaNs within the data and prints the results. This is for analysing purposes only.

        :param single_station: Wether to print empty data fields for all stations of just one.
            If False it might print out a confusing lot of data points.
        :type single_station: bool
        """
        for key in self.data:
            print(key + ": column |  # of NaN")
            for col in self.data[key].columns:
                num_nan = len(self.data[key].loc[self.data[key][col].isnull()])
                if num_nan == 0:
                    continue
                print("          " + col, end="")
                print(" "*(7-len(col)), end="|")
                print("  " + str(num_nan))
            if single_station:
                break

    def create_model_input(self, timesteps, data=None, station=None, features=None, save_data=True):
        """Shapes the data to be used for a lstm model. The output is a multidimensional array with
        the dimensions: timesteps x features x len(data) - timesteps
        The first <timesteps> values are ignored as the data points do not have enough values before
        them to fill in the number of timesteps given.

        :param timesteps: Number of values to be taken into account for a single prediction in the model
        :type timesteps: int
        :param data: The data which is used to create the model input. If none is given, self.data is used
        :type data: DataFrame, optional
        :param station: Only used if data is none. Defines the stations to consider from the whole data.
            If none, all the stations are used
        :type station: list of str, optional
        :param features: Only used if data is none. Defines the featuress to consider from the whole data.
            If none, all the features are used
        :type features: list of str, optional
        :param save_data: Either to save the data in self.model_input. This is useful as this function
            is slow if data is big
        :type save_data: bool, optional
        :return: a numpy array in the shape of: timesteps x features x len(data) - timesteps
            holding the data for the model input
        :rtype: numpy array
        """
        if features is None:
            features = self.features
        if station is None:
            station = self.station
        if data is None:
            len_data_station = len(self.data[station[0]]) - timesteps
            len_data = len_data_station * len(station)
            output = np.empty(shape=(len_data, timesteps, len(features)))
            for s in range(len(station)):
                print(station[s] + " (" + str(s+1) + "/" + str(len(station)) + ")")
                d = self.create_model_input(timesteps, self.data[station[s]], features,
                                            save_data=False)
                np.put(output, np.arange(s * len_data_station * timesteps * len(features),
                                         (s + 1) * len_data_station * timesteps * len(features)),
                       d)
            if save_data:
                self.model_input = output
            return output

        output = np.zeros(shape=(len(data) - timesteps, timesteps, len(features)))
        print("Creating model input from " + str(features))
        for i in tqdm.tqdm(range(len(data) - timesteps)):
            np.put(output, np.arange(i * timesteps * len(features),
                                     (i + 1) * timesteps * len(features)),
                   data[features].iloc[i:i + timesteps])
        if save_data:
            self.model_input = output
        return output

    def create_classes(self, num_classes, data=None, station=None,
                       features=None, one_hot_encoding=True,
                       create_classes_from_unified=True):
        """Creates classes as the model is used for classification of continuous data. Saves and
        returns num_classes with an equal number of elements out of the (unified) data from given
        features.

        :param num_classes: Number of classes to create. If more than one feature should be divided
            into classes and some features should get different number of classes, this can be a
            list of int with one number for each feature.
        :type num_classes: int or list of int
        :param data: The data to create classes for, if None it takes the data defined in station
            or the whole data from self.data
        :type data: DataFrame, optional
        :param station: The name of the station(s) from which the data should be taken if no data is
            given
        :type station: str or list of str, optional
        :param features: A list of features to create classes for. If none is given, it uses the
            self.col_to_predict
        :type features: list of str, optional
        :param one_hot_encoding: True, if each class should have its own column, false if there should
            be just one column with one number for each class, list if different for each feature
        :type one_hot_encoding: bool or list of bool
        :param create_classes_from_unified: Whether to create the groups with respect to the
            self.col_to_predict full data (True) or only the given data (False).
        :return: The data containing the given features and the newly created classes, both
            are saved in self.data as well
        :rtype: DataFrame, DataFrame
        """
        if data is None and station is None:
            data = self.unite_data(self.data)
        elif data is None:
            if type(station) == str:
                data = self.data[station]
            elif len(station) == 1:
                data = self.data[station[0]]
            else:
                data = self.unite_data(self.data, stations=station)
        if features is None:
            features = self.col_to_predict

        # The following is useful to make equally sized classes (if not unified, the
        # classes might vary more).
        if create_classes_from_unified:
            reference = self.unite_data(self.data)
        else:
            reference = data
        if type(one_hot_encoding) is not list:
            one_hot_encoding = [one_hot_encoding for x in range(len(features))]
        if type(num_classes) is not list:
            c = np.zeros((len(features)), dtype=int)
            num_classes = c + num_classes

        limits = {}
        classes = []
        for f in range(len(features)):
            limits[features[f]] = []
            for i in range(num_classes[f] + 1):
                q = 1 / num_classes[f] * i
                limits[features[f]].append(reference[features[f]].quantile(q))
                if i == 0:
                    continue
                if one_hot_encoding[f]:
                    classes.append(features[f] + "_class" + str(i - 1))
                    data[classes[-1]] = 0
                    v = 1
                else:
                    classes.append(features[f] + "_classes")
                    v = i - 1
                lower_bound = limits[features[f]][-2]
                upper_bound = limits[features[f]][-1]
                data.loc[lambda d: (d[features[f]] >= lower_bound) &
                                   (d[features[f]] < upper_bound),
                         classes[-1]] = v
        return data[features], data[classes]

    def create_wd_classes(self):
        """Creates four classes (north, east, south, west) out of the
        wd column, nw -> .5 north, .5 west, 0 south, 0 west. This removes the
        wd class from the saved data (self.data) and adds the newly created.
        """
        print("Creating multiple classes from wd (wind direction):")
        for location in tqdm.tqdm(self.data.keys()):
            for wd in ["N", "E", "S", "W"]:
                self.data[location]["wd_" + str(wd)] = self.data[location].apply(
                    # lambda x : type(x["wd"]) == str, axis=1)
                    lambda x: 1 / len(str(x["wd"])) * (str(x["wd"]).count(wd)), axis=1)

    def train_test_split(self, data, labels, timesteps, num_stations=None, test_split=.25, shuffle_data=True):
        """Creates shares from the given data for train/test separation.

        :param data: The data to separate
        :type data: numpy array of float
        :param labels: The labels for the training data
        :type labels: numpy array of float
        :param timesteps: The number of timesteps during training
        :type timesteps: int
        :param num_stations: The number of stations which are considered
        :type num_stations: int, optional
        :param test_split: The amount of testing data
        :type test_split: float, optional
        :param shuffle_data: Whether to shuffle the data
        :type shuffle_data: bool, optional
        :return: Train data, test data, train labels, test labels
        :rtype: numpy array, numpy array, numpy array, numpy array
        """
        if num_stations is None:
            num_stations = len(self.station)
        if test_split is None:
            test_split = .25
        for s in range(num_stations-1, -1, -1):
            pos = s * int(len(data)/num_stations)
            labels = labels.drop(np.arange(pos, pos+timesteps))

        if shuffle_data:
            data, labels = shuffle(data, labels)

        test_split = int(test_split*len(data))
        test_data = data[:test_split]
        train_data = data[test_split:]
        test_labels = labels[:test_split]
        train_labels = labels[test_split:]
        return train_data, test_data, train_labels, test_labels

    @staticmethod
    def cross_validation_data(data, labels, k_folds):
        """Shapes the data for k-fold cross-validation and yields the data accordingly.

        :param data: The data to divide into k folds
        :type data: numpy array of float
        :param labels: The labels matching the data
        :type labels: numpy array of float
        :param k_folds: How many folds the data should be divided to
        :type k_folds: int
        :return: k times the data to train with, the according labels, the test data, the test labels
        :rtype: numpy array, numpy array, numpy array, numpy array
        """
        for k in range(k_folds):
            start_test = int(len(data)/k_folds*k)
            end_test = int(len(data)/k_folds*(k+1))
            test_data = data[start_test:end_test]
            test_labels = labels[start_test:end_test]
            train_data = np.concatenate((data[:start_test], data[end_test:]))
            train_labels = np.concatenate((labels[:start_test], labels[end_test:]))
            yield train_data, train_labels, test_data, test_labels


def create_lstm(input_values, input_features, num_output_classes):
    """Creates and compiles the lstm model, optimized for the air pollution of Beijing dataset.

    :param input_values: Number of timesteps to consider for training and prediction
    :type input_values: int
    :param input_features: Number of features to consider for training and prediction
    :type input_features: int
    :param num_output_classes: Amount of classes for the classification
    :type num_output_classes: int
    :return: compiled tensorflow lstm model
    :rtype: keras.model
    """
    model = keras.Sequential()
    model.add(layers.LSTM(10, input_shape=(input_values, input_features), return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.LSTM(5))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(num_output_classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


def cross_validation_train(data, labels, k_fold, input_values, input_features, num_output_classes,
                           epochs=10, batch_size=128, show_plot=True, save_model=False,
                           save_model_path='models/cross_val_models/', save_model_name=None):
    """Performs cross validation on the given data.

    :param data: The data for model training and cross validation
    :type data: numpy array of float
    :param labels: The labels fitting to the data
    :type labels: numpy array of float
    :param k_fold: The amount of folds to perform
    :type k_fold: int
    :param input_values: The timesteps to consider for the lstm
    :type input_values: int
    :param input_features: The number of features the model will be fed with
    :type input_features: int
    :param num_output_classes: Amount of classes for the classification
    :type num_output_classes: int
    :param epochs: How many epochs the model will be trained
    :type epochs: int, optional
    :param batch_size: The batch size for the training
    :type batch_size: int, optional
    :param show_plot: Wether to plot the accuracy and validation accuracy for each cross validation step
    :type show_plot: bool, optional
    :param save_model: Wether to save each model after training and validation
    :type save_model: bool, optional
    :param save_model_path: The path where to save the models, only used if save_model=True
    :type save_model_path: str, optional
    :param save_model_name: The name to give each model when it is saved. Only used if
        save_model=True, default is ep<num_epochs>_k<k_fold>_<run_id>
    :type save_model_name: str, optional
    :return: A list of each models loss history and a list with each model
    :rtype: list of keras.callback.History, list of keras.model
    """
    loss_history = []
    models = []
    for i, j, k, l in tqdm.tqdm(DataHandler.cross_validation_data(data, labels, k_fold)):
        models.append(create_lstm(input_values, input_features, num_output_classes))
        print("-", end="")
        loss_history.append(models[-1].fit(i, j, epochs=epochs, batch_size=batch_size,
                                           validation_data=(k, l), verbose=1))
        if save_model:
            if save_model_name is None:
                save_model_name = f"ep{epochs}_k{k_fold}_{len(models)}"
            models[-1].save(f'{save_model_path}{save_model_name}/')

        if show_plot:
            plt.plot(loss_history[-1].history["accuracy"], label="Training Acc run " + str(len(models)))
            plt.plot(loss_history[-1].history["val_accuracy"], label="Test Acc run " + str(len(models)))
            plt.legend()
    return loss_history, models


def run_model(data, model, features=None, station=None, col_to_predict=None, num_classes=3, shuffle=False,
              timesteps=48, test_split=.25, epochs=15, batch_size=128, metrics=False):
    """Preprocesses the data and trains the given model on the data.

    :param data: A path to the data or a loaded DataHandler object
    :type data: str or pandas.DataFrame
    :param model: The compiled lstm model for the training
    :type model: keras.Model
    :param features: The features which the model will be trained on
    :type features: list of str, optional
    :param station: The station to load the data and train the model for
    :type station: str or list of str, optional
    :param col_to_predict: The columns to train the model for (default is PM2.5)
    :type col_to_predict: list of str, optional
    :param num_classes: How many classes the classification should be trained on (must fit the model)
    :type num_classes: int, optional
    :param shuffle: Whether to shuffle the data before feeding into the model
    :type shuffle: bool, optional
    :param timesteps: How many timesteps the lstm should include (must fit the model)
    :type timesteps: int, optional
    :param test_split: How much data will be used for testing and validation
    :type test_split: float, optional
    :param epochs: How many epochs the model will be trained
    :type epochs: int, optional
    :param batch_size: How big the batch size during training is
    :type batch_size: int, optional
    :param metrics: Either to return the f1 score, precision and recall as well
    :type metrics: bool, optional
    :return: Loss history of the training and the trained model. If metrics is true, it also returns
        f1 score, true and predicted labels.
    :rtype: keras.callback.History, keras.Model, if metrics is True, also float, array of int, array of int
    """
    if type(station) == str:
        station = [station]
    if type(data) == str:
        data = DataHandler(data, station=station, features_to_use=features,
                           col_to_predict=col_to_predict)
    elif type(data) != DataHandler:
        raise ValueError("Data must be either the path to the data or a DataHandler object!")

    print("---------------------Preprocessing data--------------------------")
    data.preprocess_data(minmax_features=features)
    if "wd" in features:
        features = [f for f in features + ["wd_N", "wd_E", "wd_S", "wd_W"] if f != "wd"]
    data.interpolate()
    pm_data, labels = data.create_classes(num_classes, features=["PM2.5"], station=station)

    print("-------------------Creating training data------------------------")
    if data.model_input is None or station is not None:
        data_orig = data.create_model_input(timesteps, station=station,
                                            features=features, save_data=False)
    else:
        data_orig = data.model_input

    if station is None:
        num_station = None
    else:
        num_station = len(station)
    train_data, test_data, train_labels, test_labels = data.train_test_split(
        data_orig, labels, timesteps, num_stations=num_station, test_split=test_split, shuffle_data=shuffle)

    print("---------------------Training the model--------------------------")
    loss_history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                             validation_data=(test_data, test_labels), verbose=1)

    if metrics:
        x, _, x_l, _ = data.train_test_split(data_orig, labels, timesteps, num_stations=None,
                                          test_split=0, shuffle_data=False)
        return model, loss_history, get_metrics(model, x, x_l,
                                                batch_size=batch_size,
                                                return_labels=True)
    return model, loss_history


def fed_model_fn():
    """A function for TFF to create a model during federated learning and return it is the correct type.

    :return: The LSTM model to be used as federated models
    :rtype: tff.learning.model
    """
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_fed_lstm()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=(None, 48, 12), dtype=tf.float64, name=None),
                    tf.TensorSpec(shape=(None, 3), dtype=tf.int64, name=None)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ])


def create_fed_lstm(input_values=48, input_features=12, num_output_classes=3):
    """Creates and returns a lstm model ready to be used in the federated structure

    :param input_values: Number of values to take into account for prediction ("timesteps")
    :type input_values: int, optional
    :param input_features: Number of features to take into account for prediction
    :type input_features: int, optional
    :param num_output_classes: Number of classes to do the classification for
    :type num_output_classes: int, optional
    :return: The not yet compiled tensorflow model
    :rtype: keras.Model
    """
    model = keras.Sequential()
    model.add(layers.LSTM(6,
                          return_sequences=True,
                          input_shape=(input_values, input_features),
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)
                          ))

    model.add(layers.Dropout(.3))
    model.add(layers.LSTM(4,
                          return_sequences=False,
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)
                          ))

    model.add(layers.Dropout(.3))
    model.add(layers.Dense(num_output_classes,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
                           ))

    return model

def get_metrics(model, input_data, true_labels, batch_size=128, return_labels=False):
    """Predicts values, checks them against the true labels and returns F1 and, if wanted,
    true and predicted labels. The labels can be useful for further analysing the results.

    :param model: The trained model
    :type model: keras.Model
    :param input_data: The data to predict the results from, ready to fed into the model
    :type input_data: numpy.Array of float
    :param true_labels: The true labels to compare the predictions with
    :type true_labels: numpy.Array of int
    :param batch_size: The batch size for the prediction step
    :type batch_size: int, optional
    :param return_labels: Wether to return the true and predicted labels
    :type return_labels: bool, optional
    :return: f1 score and, if return_labels is True, also the true and predicted labels
    :rtype: float and, if return_labels is True, also numpy.Array of int, numpy.Array of int
    """
    pred_labels = model.predict(input_data, batch_size=batch_size)
    pred_labels = np.argmax(pred_labels, axis=1)
    true_labels = np.argmax(np.array(true_labels), axis=1)

    # Precision and recall are equal to f1 if avg is micro, so they are not further used
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels,
                                                       average="micro")
    if return_labels:
        return f1, true_labels, pred_labels
    return f1

def save_loss_plot(loss, path):
    """Save the plot from a loss history

    :param loss: The loss history to get the data from
    :type loss: keras.callbacks.History
    :param path: The path where to save the plot
    :type path: str
    """
    plt.plot(loss.history["accuracy"], label="Training accuracy")
    plt.plot(loss.history["loss"], label="Training loss")
    plt.plot(loss.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(path)


def plot_data(data, features, interval=(0, -1), rolling_avg=None,
              show_missing_data=False, y_scale_log=False):
    """Plots the given data and PM2.5 values.
    
    :param data: The data to show
    :type data: DataHandler
    :param features: The features which should be plotted
    :type features: list of str
    :param interval: A tuple with the ID of the first and last element (-1 means the last)
    :type interval: tuple of int, optional
    :param rolling_avg: The window size for a rolling avg for PM2.5, if None is given, the
        rolling average will not be shown
    :type rolling_avg: int, optional
    :param show_missing_data: If true, the plot marks missing PM2.5 values (NaN) with a red
        background, if there are any
    :type show_missing_data: bool, optional
    :param y_scale_log: If true, the y-axis of the features (not PM2.5) will be on a log scale
    :type y_scale_log: bool, optional
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    plots = [ax.plot(data[features].iloc[interval[0]:interval[1]],
                     alpha=.5, label=features)]
    if y_scale_log:
        ax.set_yscale("log")

    ax2 = ax.twinx()
    plots.append(ax2.plot(data["PM2.5"].iloc[interval[0]:interval[1]],
                          label="PM2.5 measure", alpha=.3, color="darkred"))

    # Missing Values for PM2.5 is highlighted red
    if show_missing_data:
        interval = (0, 0)
        # print(data.loc[data["PM2.5"].isnull().iloc[startpoint:endpoint]])
        for m in data.loc[data["PM2.5"].isnull()].values:
            if m[0] <= interval[0]:
                continue
            if m[0] > interval[1] != -1:
                break
            ax.axvspan(m[0], m[0]+1, facecolor="red", alpha=.5)

    # Adding rolling average for PM2.5
    if rolling_avg is not None:
        pm_average = data["PM2.5"].iloc[interval[0]:interval[1]].rolling(window=rolling_avg).mean()
        plots.append(ax2.plot([i for i in range(interval[0], interval[0]+len(pm_average))], pm_average,
                              label="PM2.5 average", color="red"))

    # Damit alle Labels in einer Legende sind:
    pl = []
    for p in plots:
        pl += p
    labels = [l.get_label() for l in pl]
    ax.legend(pl, labels)
    
