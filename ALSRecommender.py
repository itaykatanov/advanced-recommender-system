import random
import numpy as np
import pandas as pd


class ALSRecommender():
    def __init__(self, k, regularization_rate, max_epochs, U=None, I=None, b_user=None, error_metric='rmse', b_item=None, init_biases=True):
        # Force integer in case it comes in as float
        self.k = k
        self.regularization_rate = regularization_rate
        self.max_epochs = max_epochs
        self.error_metric = error_metric
        self.n_users = None
        self.n_items = None
        self.mu = None
        self.U = U
        self.I = I
        self.b_item = b_item
        self.b_user = b_user
        self.initialized = False
        self.convergence = False
        self.init_biases = init_biases

    def calc_train_error(self, U, I, mu, b_u, b_i, R, R_selector=None):
        if R_selector is None:
            R_selector = (R > 0)
        R_hat = np.dot(U.T, I) + mu + b_u[:, None] + b_i[None, :]
        if self.error_metric == 'rmse':
            error = np.sqrt(np.sum(R_selector * pow(R_hat - R, 2)) / np.sum(R_selector))
        else:
            raise ValueError("{} is an unsupported error metric")
        return error

    def _fit_init(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col, rating_col = X.columns[:3]
        self.mu = X[rating_col].mean()
        self.R, self.user_map, self.item_map = self._get_rating_matrix(X)
        self.n_users, self.n_items = self.R.shape
        if self.init_biases:
            self.b_user = np.zeros(self.n_users)
            self.b_item = np.zeros(self.n_items)
        self.U = np.random.randn(self.k, self.n_users) / self.k
        self.I = np.random.randn(self.k, self.n_items) / self.k
        self.identity_matrix = np.eye(self.k) # (k x k)-dimensional idendity matrix
        self.epoch = 0
        self.train_errors = []
        self.initialized = True
        
    def fit(self, X, n_epochs=None):
        self._fit_init(X)
        epoch_0 = self.epoch
        if n_epochs is None:
            n_epochs = self.max_epochs - epoch_0
        # Run n_epochs iterations
        for i_epoch in range(n_epochs):
            if self.epoch >= self.max_epochs:
                print("max_epochs = {}".format(self.max_epochs))
                break
            if self.convergence:
                print("[Converged! Epoch {}/{}] train RMSE: {}".format(self.epoch, self.max_epochs, error))
                break
            # Fix I and estimate U + Estimate Bu
            for u, Ru in enumerate(self.R):
                Ru_nonzero_selector = np.nonzero(Ru)[0]
                matrix = np.zeros((self.k, self.k)) + self.regularization_rate * np.eye(self.k)
                vector = np.zeros(self.k)
                sum = 0
                for j, Rj in enumerate (Ru_nonzero_selector):
                    matrix += np.outer(self.I[:, Rj], self.I[:, Rj])
                    vector += (self.R[u, Rj] - self.b_user[u] - self.b_item[Rj] - self.mu) * self.I[:, Rj]
                    sum += (self.R[u, Rj] - self.b_item[Rj] - self.mu - self.U[:, u].dot(self.I[:,Rj]))
                self.U[:, u] = np.linalg.solve(matrix, vector)
                self.b_user[u] = sum / (len(Ru_nonzero_selector) + self.regularization_rate)
            # Fix U and estimate I + Estimate Bi
            for i, Ri in enumerate(self.R.T):
                Ri_nonzero_selector = np.nonzero(Ri)[0]
                matrix = np.zeros((self.k, self.k)) + self.regularization_rate * np.eye(self.k)
                vector = np.zeros(self.k)
                sum = 0
                for j, Rj in enumerate (Ri_nonzero_selector):
                    matrix += np.outer(self.U[:, Rj], self.U[:, Rj])
                    vector += (self.R.T[i, Rj] - self.b_item[i] - self.b_user[Rj] - self.mu) * self.U[:, Rj]
                    sum += (self.R.T[i, Rj] - self.b_user[Rj] - self.mu - self.U[:, Rj].dot(self.I[:, i]))
                self.I[:, i] = np.linalg.solve(matrix, vector)
                self.b_item[i] = sum / (len(Ri_nonzero_selector) + self.regularization_rate)
            error = self.calc_train_error(self.U, self.I, self.mu, self.b_user, self.b_item, self.R)
            self.train_errors.append(error)
            print("[Epoch {}/{}] train error: {}".format(self.epoch, self.max_epochs, error))
            self.epoch += 1
            if self.epoch > 1 and abs(error - self.train_errors[self.epoch - 2]) < 0.001:
                self.convergence = True;
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        X['rating_baseline'] = self.mu
        X['rating'] = 0
        known_user_and_item_mask = (X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index))
        X_known, X_unknown = X[known_user_and_item_mask], X[~known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        rating_pred = np.array([np.sum(self.U[:, u_ind] * self.I[:, i_ind]) for u_ind, i_ind in zip(user_inds, item_inds)])
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        min_rating = np.min(self.R[np.nonzero(self.R)])
        max_rating = np.max(self.R)
        X.loc[X['rating'] < min_rating, 'rating'] = min_rating
        X.loc[X['rating'] > max_rating, 'rating'] = max_rating
        return X['rating'].values

    def _get_rating_matrix(self, X):
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


def random_search():
    """Random search for hyperparameter optimization"""
    MAX_EVALS = 1
    max_epochs = 2
    model_list = []
    results = pd.DataFrame(columns = ['RMSE Score', 'MAE Score', 'R^2', 'Model'], index = list(range(MAX_EVALS)))
    k_list = [5, 10, 15, 20, 30, 50, 60, 75, 90, 110, 130, 150]
    for i in range(MAX_EVALS):
        k = random.sample(k_list, k=1).pop(0)
        regularization_rate = np.random.uniform(0.001, 0.1)
        print("Running with k=%d, regularizationrate=%f" % (k, regularization_rate))
        model = ALSRecommender(k=k, regularization_rate=regularization_rate, max_epochs=max_epochs)
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

train_df = pd.read_csv('Train.csv', names=['userId', 'movieId', 'rating'])
validation_df = pd.read_csv('Validation.csv', names=['userId', 'movieId', 'rating'])
train_and_validation_df = pd.concat([train_df, validation_df])
result = random_search()
chosenModel = result[0]
print("Final model test RMSE is: " + str(chosenModel.test_rmse))
print("Final model test MAE is: " + str(chosenModel.test_mae))
print("Final model test R^w is: " + str(chosenModel.test_r_squared))
print("Staring to train Final model..")
finalModel = ALSRecommender(k=chosenModel.k, U=chosenModel.U, I=chosenModel.I, b_user=chosenModel.b_user, b_item=chosenModel.b_item, regularization_rate=chosenModel.regularization_rate, max_epochs=2)
finalModel.fit(train_and_validation_df)
test_df = pd.read_csv('Test.csv', names=['User_ID_Alias', 'Movie_ID_Alias', 'Rating'])
predictions = finalModel.predict(test_df[['User_ID_Alias', 'Movie_ID_Alias']])
test_df['Rating'] = predictions
export_csv = test_df.to_csv (r'C:\Users\itay.katanov\Desktop\איתי פרטי\תואר שני\שנה ב סמסטר א\מערכות המלצה מתקדמות\HW1\RecSystem_ex1\ALS_301406641_301796447.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

print("Done Kapara")