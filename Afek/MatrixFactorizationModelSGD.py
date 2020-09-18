import numpy as np

from optimization_objects import LearningRateScheduler, SgdEarlyStopping
from matrix_factorization_abstract import MatrixFactorizationWithBiases
from momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D
from config import MEASURE


class MatrixFactorizationWithBiasesSGD(MatrixFactorizationWithBiases):
    # initialization of model's parameters
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension, config.print_metrics)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.lr = config.lr
        self.early_stopping = None
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_users_bias = config.l2_users_bias
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.number_bias_epochs = config.bias_epochs
        self.beta = config.beta
        self.results = {}
        self.users_h_gradient = None
        self.items_h_gradient = None
        self.user_biases_gradient = None
        self.item_biases_gradient = None

    # initialization of model's weights
    def weight_init(self, user_map, item_map, global_bias):
        self.global_bias = global_bias
        self.user_map, self.item_map = user_map, item_map
        self.U = np.random.normal(scale=0.2 / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=0.2 / self.h_len, size=(self.n_items, self.h_len))
        self.users_h_gradient = MomentumWrapper2D(self.n_users, self.h_len, self.beta)
        self.items_h_gradient = MomentumWrapper2D(self.n_items, self.h_len, self.beta)
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.user_biases_gradient = MomentumWrapper1D(self.n_users, self.beta)
        self.item_biases_gradient = MomentumWrapper1D(self.n_items, self.beta)

    def fit(self, train, user_map: dict, item_map: dict, validation=None):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.early_stopping = SgdEarlyStopping()
        self.lr = LearningRateScheduler(self.lr)
        train = train.values
        self.weight_init(user_map, item_map, np.mean(train[:, 2]))
        validation_error = None
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(train)
            self.run_epoch(train, epoch)
            # calculate train/validation error and loss
            train_accuracy = self.prediction_error(train, MEASURE)
            train_loss = self.calc_loss(train)
            convergence_params = {'train_accuracy': train_accuracy, 'train_loss': train_loss}
            if validation is not None:
                validation_error = self.prediction_error(validation, MEASURE)
                validation_loss = self.calc_loss(validation)
                if self.early_stopping.stop(self, epoch, validation_error):
                    break
                convergence_params.update({'validation_accuracy': validation_error, 'validation_loss': validation_loss})
            self.record(epoch, **convergence_params)
        return validation_error

    def run_epoch(self, data, epoch):
        lr = self.lr.update(epoch)
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            error = rating - prediction
            u_b_gradient = (error - self.l2_users_bias * self.user_biases[user])
            i_b_gradient = (error - self.l2_items_bias * self.item_biases[item])
            self.user_biases[user] += lr * self.user_biases_gradient.get(u_b_gradient, user)
            self.item_biases[item] += lr * self.item_biases_gradient.get(i_b_gradient, item)
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                self.U[user, :] += lr * self.users_h_gradient.get(u_grad, user)
                self.V[item, :] += lr * self.items_h_gradient.get(v_grad, item)
