import numpy as np
import pandas as pd


class MatrixFactorizationWithBiases:
    def __init__(self, seed, hidden_dimension, print_metrics=True):
        self.h_len = hidden_dimension
        self.results = {}
        np.random.seed(seed)
        self.print_metrics = print_metrics
        self.user_map = None
        self.item_map = None
        self.global_bias = None
        self.user_biases = None
        self.item_biases = None
        self.U = None
        self.V = None
        self.l2_users_bias = None
        self.l2_items_bias = None
        self.l2_users = None
        self.l2_items = None

    def get_results(self):
        return pd.DataFrame.from_dict(self.results)

    def record(self, epoch, **kwargs):
        if self.print_metrics:
            print(f"epoch # {epoch} : \n")
        for key, value in kwargs.items():
            key = f"{key}"
            if not self.results.get(key):
                self.results[key] = []
            self.results[key].append(value)
            if self.print_metrics:
                print(f"{key} : {np.round(value, 5)}")

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, train: pd.DataFrame, user_map: dict, item_map: dict, validation: np.array = None):
        pass

    def predict(self, user, item):
        """
        predict on user and item with their original ids not internal ids
        """
        user = self.user_map.get(user, None)
        item = self.item_map.get(item, None)
        # TODO: remove this check
        if (user is None) or (item is None):
            print('item or user is none')
        if user:
            if item:
                prediction = self.predict_on_pair(user, item)
            else:
                prediction = self.global_bias + self.user_biases[user]
        else:
            if item:
                prediction = self.global_bias + self.item_biases[item]
            else:
                prediction = self.global_bias
        return np.clip(prediction, 1, 5)

    def calc_loss(self, x):
        loss = 0
        parameters = [self.user_biases, self.item_biases, self.U, self.V]
        regularizations = [self.l2_users_bias, self.l2_items_bias, self.l2_users, self.l2_items]
        for i in range(len(parameters)):
            loss += regularizations[i] * np.sum(np.square(parameters[i]))
        return loss + self.prediction_error(x, 'squared_error')

    def prediction_error(self, x, measure_function="rmse"):
        error_functions = {'rmse': np.square, 'mse': np.square, 'squared_error': np.square, 'mae': np.abs,
                           'r2': np.square}
        error_function = error_functions[measure_function]
        e = 0
        t = 0
        for row in x:
            user, item, rating = row
            e += error_function(rating - self.predict_on_pair(user, item))
            if measure_function == 'r2':
                t += error_function(rating - self.global_bias)
        if measure_function == 'mse':
            return e / x.shape[0]
        elif measure_function == 'rmse':
            return np.sqrt(e / x.shape[0])
        elif measure_function == 'squared_error':
            return e
        elif measure_function == 'r2':
            return 1 - e / t
        else:
            return e / x.shape[0]

    def predict_on_pair(self, user, item):
        return np.clip(self.global_bias + self.user_biases[user] + self.item_biases[item] \
                       + self.U[user, :].dot(self.V[item, :].T), 1, 5)
