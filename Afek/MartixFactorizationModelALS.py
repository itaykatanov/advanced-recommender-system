import numpy as np

from matrix_factorization_abstract import MatrixFactorizationWithBiases
from optimization_objects import AlsEarlyStopping
from config import MEASURE


class MatrixFactorizationWithBiasesALS(MatrixFactorizationWithBiases):
    # initialization of model's parameters
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension, config.print_metrics)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_users_bias = config.l2_users_bias
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.early_stopping = None
        self.number_bias_epochs = config.bias_epochs
        self.user_dict = {}
        self.item_dict = {}
        self.results = {}

    # initialization of model's weights
    def weight_init(self, user_map, item_map):
        self.user_map, self.item_map = user_map, item_map
        self.U = np.random.normal(scale=1. / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=1. / self.h_len, size=(self.n_items, self.h_len))
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

    def dict_init(self, df):
        for user in range(self.n_users):  # get for current user all ratings, and items indices
            self.user_dict[user] = {'items': df[df.user == user]['item'].values,
                                    'ratings': df[df.user == user]['Ratings_Rating'].values}
        for item in range(self.n_items):  # get for current item all ratings, and users indices
            self.item_dict[item] = {'users': df[df.item == item]['user'].values,
                                    'ratings': df[df.item == item]['Ratings_Rating'].values}

    def als_step(self):
        # users
        for u in range(self.n_users):
            # user hidden
            item_vecs = self.V[self.user_dict[u]['items'], :]
            item_biases, user_bias = self.item_biases[self.user_dict[u]['items']], self.user_biases[u]
            ratings = self.user_dict[u]['ratings']
            n_items = ratings.size
            error = ratings - item_biases - user_bias - self.global_bias
            left_matrix = np.linalg.inv(item_vecs.T.dot(item_vecs) + np.eye(self.h_len) * self.l2_users)
            right_vec = np.multiply(item_vecs.T, error).sum(axis=1)
            self.U[u, :] = left_matrix.dot(right_vec)  # update current user low dimensional vector
            # user bias
            right_hand_side = np.sum(ratings - item_biases - self.global_bias - item_vecs.dot(self.U[u, :]))
            left_hand_side = 1 / (self.l2_users_bias + n_items)
            self.user_biases[u] = left_hand_side * right_hand_side  # update current user biases
        # items
        for i in range(self.n_items):
            # item hidden
            user_vecs = self.U[self.item_dict[i]['users'], :]
            user_biases, item_bias = self.user_biases[self.item_dict[i]['users']], self.item_biases[i]
            ratings = self.item_dict[i]['ratings']
            n_users = ratings.size
            error = ratings - user_biases - item_bias - self.global_bias
            left_matrix = np.linalg.inv(user_vecs.T.dot(user_vecs) + np.eye(self.h_len) * self.l2_items)
            right_vec = np.multiply(user_vecs.T, error).sum(axis=1)
            self.V[i, :] = left_matrix.dot(right_vec)  # update current item low dimensional vector
            # item bias
            right_hand_side = np.sum(ratings - user_biases - self.global_bias - user_vecs.dot(self.V[i, :]))
            left_hand_side = 1 / (self.l2_items_bias + n_users)
            self.item_biases[i] = left_hand_side * right_hand_side  # update current item biases

    def fit(self, train, user_map: dict, item_map: dict, validation=None):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.early_stopping = AlsEarlyStopping()
        train = train.sort_values(by=['user', 'item'])
        # validation = validation.sort_values(by=['user', 'item'])
        self.dict_init(train)
        train = train.values
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        validation_error = None
        for epoch in range(1, self.epochs + 1):
            self.als_step()
            # calculate train/validation error and loss
            train_accuracy = self.prediction_error(train, MEASURE)
            train_loss = self.calc_loss(train)
            convergence_params = {'train_accuracy': train_accuracy, 'train_loss': train_loss}
            if validation is not None:
                validation_error = self.prediction_error(validation, MEASURE)
                validation_loss = self.calc_loss(validation)
                if self.early_stopping.stop(epoch, validation_error):
                    break
                convergence_params.update({'validation_accuracy': validation_error, 'validation_loss': validation_loss})
            self.record(epoch, **convergence_params)
        return validation_error

    def fit_all(self, train, user_map: dict, item_map: dict):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.early_stopping = AlsEarlyStopping()
        train = train.sort_values(by=['user', 'item'])
        self.dict_init(train)
        train = train.values
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        for epoch in range(1, self.epochs + 1):
            self.als_step()
            # calculate train/validation error and loss
            train_error = self.prediction_error(train, MEASURE)
            self.record(epoch, train_accuracy=train_error,
                        train_loss=self.calc_loss(train))
            if self.early_stopping.stop(epoch, train_error):
                break
        print(f"train_final_score: {train_error}")
        return train_error
