import numpy as np
import pandas as pd
import skopt
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence

from config import HYPER_PARAM_SEARCH, CHECKPOINT_NAME, \
    HYPER_PARAM_SEARCH_N_ITER, HYPER_PARAM_FILE_NAME, TRAIN_PATH, VALIDATION_PATH, space, SEED, TEST_PATH, \
    FIT_ON_TRAIN_VALIDATION, USERS_COL_NAME, ITEMS_COL_NAME
from utils import preprocess_for_mf, get_mf


@skopt.utils.use_named_args(space)
def objective(**params):
    mf.set_params(**params)
    print({i: np.round(v, 3) for i, v in mf.__dict__.items() if i in params.keys()})
    return mf.fit(train, user_map, item_map, validation)


def train_on_all_predict_on_test(train, validation, test):
    train_and_validation = pd.concat([train, validation], ignore_index=True)
    train_and_validation, user_map, item_map = preprocess_for_mf(train_and_validation)
    n_users, n_items = len(user_map), len(item_map)
    models = get_mf(n_users, n_items,True)
    model_names = ['sgd', 'als']
    for i, mf in enumerate(models):
        mf.fit(train_and_validation, user_map, item_map)
        predictions = [mf.predict(int(row[USERS_COL_NAME]), int(row[ITEMS_COL_NAME])) for index, row in test.iterrows()]
        test[F"prediction_{model_names[i]}"] = predictions
    test.to_csv('submission_results.csv', index = False)


def run_exp(model, train, user_map, item_map, validation):
    if HYPER_PARAM_SEARCH:
        checkpoint_saver = CheckpointSaver(CHECKPOINT_NAME)
        res_gp = gp_minimize(objective, space, n_calls=HYPER_PARAM_SEARCH_N_ITER, random_state=SEED,
                             callback=[checkpoint_saver])
        skopt.dump(res_gp, HYPER_PARAM_FILE_NAME, store_objective=False)
        plot_convergence(res_gp)
    else:
        model.fit(train, user_map, item_map, validation)


if __name__ == '__main__':
    train, validation, test = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH), pd.read_csv(TEST_PATH)
    if FIT_ON_TRAIN_VALIDATION:
        # Final Run on all of the train data
        train_on_all_predict_on_test(train, validation, test)
    else:
        train, validation, user_map, item_map = preprocess_for_mf(train, validation)
        validation = validation.values
        n_users, n_items = len(user_map), len(item_map)
        mf = get_mf(n_users, n_items)
        run_exp(mf, train, user_map, item_map, validation)
