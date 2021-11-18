import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import time

import plotly.graph_objects as go

import src.features.make_sessions as make_sessions
from src.models.performance_metrics import show_results


class ModelFullyConnectedNN:

    def __init__(self):

        self.version = 'fully_connected_nn__' + datetime.datetime.today().strftime("%Y%m%d")

        self.global_config = {
            'show_plot': False,
            'save_plot': True
        }

        self.vardict = self.get_model_vardict()

        self.impute_missing_variables = dict()

        self.model_config = {
            'nb_sessions': -1
            , 'batch_size': 32
            , 'num_epochs': 10
            , 'input_shape_test': (-1,)
        }

        self.model = dict()
        self.model_metric = {
            'training': dict(),
            'test': dict()
        }

        self.time_for_training = 0

    def define_model(self):

        self.model['neural_network'] = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(self.model_config['nb_sessions'],)),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(2),
            ]
        )

        self.model['loss_object'] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model['optimizer'] = tf.keras.optimizers.Adam()

    def define_metric(self):

        self.model_metric['training']['loss_train'] = {
            'training_or_test': 'training',
            'metric': tf.keras.metrics.Mean(name="loss_train"),
            'results': []
        }
        self.model_metric['training']['accuracy_train'] = {
            'training_or_test': 'training',
            'metric': tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_train"),
            'results': []
        }
        self.model_metric['test']['loss_test'] = {
            'training_or_test': 'test',
            'metric': tf.keras.metrics.Mean(name="loss_test"),
            'results': []
        }
        self.model_metric['test']['accuracy_test'] = {
            'training_or_test': 'test',
            'metric': tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_test"),
            'results': []
        }

    def get_model_vardict(self):

        vardict = dict()

        return vardict

    def preprocessing_training(self, dataset):

        self.model_config['nb_sessions'] = max(dataset["id_session"]) + 1

        dataset_word_language_sessions = (
            dataset.groupby(["id_vocab", "language_asked"])
                .apply(lambda x: make_sessions.define_word_language_sessions(x, self.model_config['nb_sessions']))
                .reset_index()
        )

        dataset_word_language_sessions_multiplied = (
            dataset_word_language_sessions.groupby(["id_vocab", "language_asked"])
                .apply(make_sessions.multiply_word_language_sessions)
                .reset_index()
        )
        del dataset_word_language_sessions_multiplied['level_2']

        self.model_config['nb_sessions'] = dataset_word_language_sessions_multiplied["before"].map(len).max()

        # add None to complete sessions
        dataset_word_language_sessions_multiplied["sessions_standardized"] = (
            dataset_word_language_sessions_multiplied["before"].map(
                lambda x: make_sessions.standardize_sessions(x, self.model_config['nb_sessions'])
            )
        )

        dataset_word_language_sessions_multiplied["sessions_numeric"] = (
            dataset_word_language_sessions_multiplied["sessions_standardized"].map(
                make_sessions.map_session_to_numeric
            )
        )

        return dataset_word_language_sessions_multiplied

    def preparation_nn(self, dataset):

        # transform to array of array
        sessions = np.array(dataset["sessions_numeric"].tolist(), dtype="int8")
        print('Training shape - {}'.format(sessions.shape))

        targets = np.array(dataset["result"].tolist(), dtype="int8")  # to try: bool
        print('Training shape - {}'.format(targets.shape))

        return sessions, targets

    def create_dataset_training(self, dataset):

        sessions, targets = self.preparation_nn(dataset)

        ds_train = (
            tf.data.Dataset.from_tensor_slices((sessions, targets))
                .shuffle(buffer_size=10 * self.model_config['batch_size'])
                .repeat(self.model_config['num_epochs'])
                .batch(self.model_config['batch_size'])
        )

        return ds_train

    @tf.function
    def step_training(self, sessions, labels):

        with tf.GradientTape() as tape:
            predictions = self.model['neural_network'](sessions, training=True)
            loss = self.model['loss_object'](labels, predictions)

        gradients = tape.gradient(loss, self.model['neural_network'].trainable_variables)
        self.model['optimizer'].apply_gradients(zip(gradients, self.model['neural_network'].trainable_variables))

        self.model_metric['training']['loss_train']['metric'](loss)
        self.model_metric['training']['accuracy_train']['metric'](labels, predictions)

    def create_dataset_test(self, dataset_test):

        sessions_test, targets_test = self.preparation_nn(dataset_test)

        ds_test = tf.data.Dataset.from_tensor_slices((sessions_test, targets_test)).batch(
            self.model_config['batch_size']
        )

        return ds_test

    @tf.function
    def step_test(self, sessions, labels):

        predictions = self.model['neural_network'](sessions, training=False)
        t_loss = self.model['loss_object'](labels, predictions)

        self.model_metric['test']['loss_test']['metric'](t_loss)
        self.model_metric['test']['accuracy_test']['metric'](labels, predictions)

    def train(self, dataset, dataset_test=None):

        ds_train = self.create_dataset_training(dataset)
        if dataset_test is not None:
            ds_test = self.create_dataset_test(dataset_test)

        self.define_model()
        self.define_metric()

        start = time.time()

        for epoch in range(self.model_config['num_epochs']):

            # Reset the metrics at the start of the next epoch
            for metric in self.model_metric['training'].keys():
                self.model_metric['training'][metric]['metric'].reset_states()
            if dataset_test is not None:
                for metric in self.model_metric['test'].keys():
                    self.model_metric['test'][metric]['metric'].reset_states()

            for sessions, labels in ds_train:
                self.step_training(sessions, labels)

            for metric in self.model_metric['training'].keys():
                self.model_metric['training'][metric]["results"].append(self.model_metric['training'][metric]["metric"].result())

            if dataset_test is not None:
                for sessions_test, labels_test in ds_test:
                    self.step_test(sessions_test, labels_test)
                for metric in self.model_metric['test']:
                    self.model_metric['test'][metric]["results"].append(self.model_metric['test'][metric]["metric"].result())

            self.print_epoch_result(epoch, dataset_test)

        end = time.time()

        self.time_for_training = end - start

    def print_epoch_result(self, epoch, dataset_test):

        print("Epoch {:3d}".format(epoch + 1))

        for metric in self.model_metric['training'].keys():
            print("{}: {}".format(metric, self.model_metric['training'][metric]["results"][-1]))

        if dataset_test is not None:
            for metric in self.model_metric['test'].keys():
                print("{}: {}".format(metric, self.model_metric['test'][metric]["results"][-1]))

    def plot_loss(self):

        # Create traces
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.model_metric['training']['loss_train']["results"]))),
                y=self.model_metric['training']['loss_train']["results"],
                mode="lines",
                name="loss_train",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.model_metric['test']['loss_test']["results"]))),
                y=self.model_metric['test']['loss_test']["results"],
                mode="lines",
                name="loss_test",
            )
        )

        fig.update_layout(
            title="Loss train vs test per epoch",
            xaxis_title="epoch",
            yaxis_title="loss",
            yaxis={'rangemode': 'tozero'},
            legend={"itemsizing": "constant"},
        )

        fig.show()

    def plot_accuracy(self):

        # Create traces
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.model_metric['training']['accuracy_train']["results"]))),
                y=self.model_metric['training']['accuracy_train']["results"],
                mode="lines",
                name="accuracy_train",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.model_metric['test']['accuracy_test']["results"]))),
                y=self.model_metric['test']['accuracy_test']["results"],
                mode="lines",
                name="accuracy_test",
            )
        )

        fig.update_layout(
            title="Accuracy train vs test per epoch",
            xaxis_title="epoch",
            yaxis_title="accuracy",
            legend={"itemsizing": "constant"},
        )

        fig.update_yaxes(
            range=[0, 1]
        )

        fig.show()

    def preprocessing_inference(self, dataset, dataset_historical):

        nb_sessions_inference = (
                max(dataset["id_session"]) + 1
        )  # because it exists id_session = 0

        dataset_historical = dataset_historical.copy().append(dataset)

        dataset_word_language_sessions_valid = (
            dataset_historical.groupby(["id_vocab", "language_asked"])
                .apply(lambda x: make_sessions.define_word_language_sessions(x, nb_sessions_inference))
                .reset_index()
        )

        dataset = dataset[
            dataset.groupby(["id_vocab", "language_asked"])["id_session"].transform(max)
            == dataset["id_session"]
            ]

        dataset = pd.merge(
            dataset,
            dataset_word_language_sessions_valid,
            on=["id_vocab", "language_asked"],
        )

        dataset = (
            dataset.groupby(["id_vocab", "language_asked"])
                .apply(make_sessions.separate_before_result)
                .reset_index()
        )
        del dataset['level_2']

        # add None to complete sessions
        dataset["sessions_standardized"] = dataset["before"].map(
            lambda x: make_sessions.standardize_sessions(x, self.model_config['nb_sessions'])
        )

        dataset["sessions_numeric"] = dataset["sessions_standardized"].map(
            make_sessions.map_session_to_numeric
        )

        return dataset

    def predict(self, dataset, target_present=False):

        X_valid = dataset[self.vardict["into_model"]].copy()

        predictions = X_valid.copy()
        predictions["y_pred"] = self.model.predict(X_valid)
        predictions["y_proba"] = [x[1] for x in self.model.predict_proba(X_valid)]

        if target_present:
            predictions["y_true"] = dataset[self.vardict["target"]].copy()

        return predictions

    def predict_and_show_results(self, dataset_valid, save_folder="data/processed"):

        y_valid = dataset_valid[self.vardict["target"]].copy()
        dataset_valid = self.preprocessing_inference(dataset_valid)
        predictions = self.predict(dataset=dataset_valid, target_present=False)
        predictions["y_true"] = y_valid.values.tolist()

        show_results(
            predictions,
            model_name=self.version,
            show_plot=self.global_config['show_plot'],
            save_plot=self.global_config['save_plot'],
            save_folder=save_folder,
        )
