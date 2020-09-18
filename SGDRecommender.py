import math
import random
import numpy as np
import pandas as pd

class SGDRecommender():
    def __init__(self, k, U=None, I=None, b_user=None, b_item=None, learning_rate=0.1, max_epochs=15, error_metric='rmse',user_reg=0.01, item_reg=0.01, user_bias_reg=0.01, item_bias_reg=0.01, init_biases=True):
        self.k = k
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_metric = error_metric
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg
        self.init_biases = init_biases

        self.U = U
        self.I = I
        self.b_u = b_user
        self.b_i = b_item
        self.test_rmse = None
        self.test_mae = None
        self.test_r_squared = None
        self.convergence = False
        self.initialized = False

    def calc_train_error(self, U, I, mu, b_u, b_i, R, R_selector=None):
        if R_selector is None:
            R_selector = (R > 0)
        R_hat = np.dot(U, I.T) + mu + b_u[:, None] + b_i[None, :]
        if self.error_metric == 'rmse':
            error = np.sqrt(np.sum(R_selector * pow(R_hat - R, 2)) / np.sum(R_selector))
        else:
            raise ValueError("{} is an unsupported error metric")
        return error

    def _fit_init(self, X, init_biases):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        user_col, item_col, rating_col = X.columns[:3]
        self.mu = X[rating_col].mean()
        self.R, self.user_map, self.item_map = self._get_rating_matrix(X)
        n_users, n_items = self.R.shape
        if init_biases:
            self.b_u = np.zeros(n_users)
            self.b_i = np.zeros(n_items)
        self.U = np.random.normal(scale=1.0/self.k, size=(n_users, self.k))
        self.I = np.random.normal(scale=1.0/self.k, size=(n_items, self.k))
        self.epoch = 0
        self.train_errors = []
        self.initialized = True

    def fit(self, X, n_epochs=None, init_biases=True):
        X = X.copy()
        # Allow continuation from previous state if n_epochs is given. Otherwise start from scratch.
        if n_epochs is None:
            self.initialized = False
        if not self.initialized:
            self._fit_init(X, init_biases)
        X.iloc[:, 0] = X.iloc[:, 0].map(self.user_map)
        X.iloc[:, 1] = X.iloc[:, 1].map(self.item_map)
        epoch_0 = self.epoch
        if n_epochs is None:
            n_epochs = self.max_epochs - epoch_0
        n_users, n_items = self.R.shape

        # Repeat until convergence
        for i_epoch in range(n_epochs):
            print("[Start Epoch {}/{}]".format(self.epoch, self.max_epochs))
            if self.convergence:
                print("[Converged! Epoch {}/{}] train RMSE: {}".format(self.epoch, self.max_epochs, error))
                break
            # Shuffle X
            X = X.sample(frac=1)
            for row in X.itertuples():
                index, user, item, rating = row[:4]
                pred = self.predict_1_train(user, item)
                err = self.R[user, item] - pred
                self.b_u[user] += self.learning_rate * (err - self.user_bias_reg * self.b_u[user])
                self.b_i[item] += self.learning_rate * (err - self.item_bias_reg * self.b_i[item])
                self.U[user, :] += self.learning_rate * (err * self.I[item, :] - self.user_reg * self.U[user, :])
                self.I[item, :] += self.learning_rate * (err * self.U[user, :] - self.item_reg * self.I[item, :])
            error = self.calc_train_error(self.U, self.I, self.mu, self.b_u, self.b_i, self.R)
            if error == 0 or math.isnan(error):
                print("ERROORORRRRRROROROOR!!!!!!!!!!!!!!!!!!")
                break
            self.train_errors.append(error)
            print("[Epoch {}/{}] train RMSE: {}".format(self.epoch, self.max_epochs, error))
            self.epoch += 1
            if self.epoch > 1 and abs(error - self.train_errors[self.epoch - 2]) < 0.001:
                self.convergence = True;
        return self

    def predict_1_train(self, user, item):
        pred = self.mu + self.b_u[user] + self.b_i[item]
        pred += np.dot(self.U[user, :], self.I[item, :])
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1
        return pred

    def _get_rating_matrix(self, X):
        """Function to generate a ratings matrx and mappings for
        the user and item ids to the row and column indices
        Parameters
        ----------
        X : pandas.DataFrame, shape=(n_ratings,>=3)
            First 3 columns must be in order of user, item, rating.
        Returns
        -------
        rating_matrix : 2d numpy array, shape=(n_users, n_items)
        user_map : pandas Series, shape=(n_users,)
            Mapping from the original user id to an integer in the range [0,n_users)
        item_map : pandas Series, shape=(n_items,)
            Mapping from the original item id to an integer in the range [0,n_items)
        """
        user_col, item_col, rating_col = X.columns[:3]
        rating = X[rating_col]
        user_map = pd.Series(
            index=np.unique(X[user_col]),
            data=np.arange(X[user_col].nunique()),
            name='user_map',
        )
        item_map = pd.Series(
            index=np.unique(X[item_col]),
            data=np.arange(X[item_col].nunique()),
            name='columns_map',
        )
        user_inds = X[user_col].map(user_map)
        item_inds = X[item_col].map(item_map)
        rating_matrix = (
            pd.pivot_table(
                data=X,
                values=rating_col,
                index=user_inds,
                columns=item_inds,
            )
                .fillna(0)
                .values
        )
        return rating_matrix, user_map, item_map

    def predict(self, X):
        """Generate predictions for user/item pairs

        Parameters
        ----------
        X : pandas dataframe, shape = (n_pairs, 2)
            User, item dataframe

        Returns
        -------
        rating_pred : 1d numpy array, shape = (n_pairs,)
            Array of rating predictions for each user/item pair
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        known_user_and_item_mask = (X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index))
        X_known = X[known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        rating_pred = np.array([self.predict_1_train(u_ind, i_ind) for u_ind, i_ind in zip(user_inds, item_inds)])
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        return X['rating'].values


def random_search():
    """Random search for hyperparameter optimization"""
    MAX_EVALS = 20
    max_epochs = 20
    model_list = []
    results = pd.DataFrame(columns = ['RMSE Score', 'MAE Score', 'R^2', 'Model'], index = list(range(MAX_EVALS)))
    k_list = [5, 10, 15, 20, 30, 50]

    for i in range(MAX_EVALS):
        k = random.sample(k_list, k=1).pop(0)
        learning_rate = np.random.uniform(0.001, 0.05)
        user_reg = np.random.uniform(0.001, 0.05)
        item_reg = np.random.uniform(0.001, 0.05)
        user_bias_reg = np.random.uniform(0.001, 0.05)
        item_bias_reg = np.random.uniform(0.001, 0.05)
        print("Running with k=%d, learning rate=%f,  user reg=%f, item reg=%f, user bias reg=%f, item bias reg=%f"
              % (k, learning_rate, user_reg, item_reg, user_bias_reg, item_bias_reg))
        model = SGDRecommender(k=k, learning_rate=learning_rate, user_reg=user_reg, item_reg=item_reg, user_bias_reg=user_bias_reg,
                               item_bias_reg=item_bias_reg, max_epochs=max_epochs, error_metric='rmse')
        model.fit(train_df)
        predictions = model.predict(validation_df[['userId', 'movieId']])
        test_RMSE = np.sqrt(np.sum(pow(predictions - validation_df['rating'], 2)) / validation_df['rating'].__len__())
        test_MAE = np.sum(abs(predictions - validation_df['rating'])) / validation_df['rating'].__len__()
        test_R_squared =  1 - (pow(test_RMSE, 2) / np.var(validation_df['rating']))
        model.test_rmse = test_RMSE
        model.test_mae = test_MAE
        model.test_r_squared = test_R_squared
        print("test RMSE is: " + str(test_RMSE))
        print("test MAE is: " + str(test_MAE))
        print("test R^2 is: " + str(test_R_squared))
        model_list.append(model)
        results.loc[i, ['RMSE Score']] = test_RMSE
        results.loc[i, ['MAE Score']] = test_MAE
        results.loc[i, ['R^2']] = test_R_squared
        results.loc[i, ['Model']] = model
    # Sort with best score on top
    results.sort_values('RMSE Score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    sorted_model_list = sorted(model_list, key=lambda x: x.test_rmse)
    return sorted_model_list


train_df = pd.read_csv('train.csv', names=['userId', 'movieId', 'rating'])
validation_df = pd.read_csv('Validation.csv', names=['userId', 'movieId', 'rating'])
train_and_validation_df = pd.concat([train_df, validation_df])
result = random_search()
chosenModel = result[0]
print("Final model test RMSE is: " + str(chosenModel.test_rmse))
print("Final model test MAE is: " + str(chosenModel.test_mae))
print("Final model test R^2 is: " + str(chosenModel.test_r_squared))
print("Staring to train Final model..")
finalModel = SGDRecommender(k=chosenModel.k, U=chosenModel.U, I=chosenModel.I, b_user=chosenModel.b_u, b_item=chosenModel.b_i,
                            learning_rate=chosenModel.learning_rate, user_reg=chosenModel.user_reg,
                            item_reg=chosenModel.item_reg, user_bias_reg=chosenModel.user_bias_reg,
                            item_bias_reg=chosenModel.item_bias_reg, max_epochs=30, error_metric='rmse', init_biases=False)
finalModel.fit(train_and_validation_df)
test_df = pd.read_csv('Test.csv', names=['User_ID_Alias', 'Movie_ID_Alias', 'Rating'])
predictions = finalModel.predict(test_df[['User_ID_Alias', 'Movie_ID_Alias']])
test_df['Rating'] = predictions
export_csv = test_df.to_csv (r'C:\Users\itay.katanov\Desktop\איתי פרטי\תואר שני\שנה ב סמסטר א\מערכות המלצה מתקדמות\HW1\RecSystem_ex1\SGD_301406641_301796447.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
print("Done Kapara")

