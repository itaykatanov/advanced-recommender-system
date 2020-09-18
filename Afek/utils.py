import pandas as pd

from MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from config import USER_COL, ITEM_COL, USERS_COL_NAME, ITEMS_COL_NAME, RATING_COL_NAME, SGD_CONFIG, SGD, ALS_CONFIG


def preprocess_for_mf(train, validation=None):
    train[USER_COL] = pd.factorize(train[USERS_COL_NAME])[0]
    train[ITEM_COL] = pd.factorize(train[ITEMS_COL_NAME])[0]
    user_map = train[[USER_COL, USERS_COL_NAME]].drop_duplicates()
    user_map = user_map.set_index(USERS_COL_NAME).to_dict()[USER_COL]
    item_map = train[[ITEM_COL, ITEMS_COL_NAME]].drop_duplicates()
    item_map = item_map.set_index(ITEMS_COL_NAME).to_dict()[ITEM_COL]
    cols_to_use = [USER_COL, ITEM_COL, RATING_COL_NAME]
    if validation is not None:
        validation = validation[
            validation[USERS_COL_NAME].isin(train[USERS_COL_NAME].unique()) & validation[ITEMS_COL_NAME].isin(
                train[ITEMS_COL_NAME].unique())]
        validation[USER_COL] = validation[USERS_COL_NAME].map(user_map)
        validation[ITEM_COL] = validation[ITEMS_COL_NAME].map(item_map)
        return train[cols_to_use], validation[cols_to_use], user_map, item_map
    return train[cols_to_use], user_map, item_map


def get_mf(users_len, items_len, return_both=False):
    sgd_config = SGD_CONFIG
    als_config = ALS_CONFIG
    sgd_config.add_attributes(n_users=users_len, n_items=items_len)
    als_config.add_attributes(n_users=users_len, n_items=items_len)
    if return_both:
        return [MatrixFactorizationWithBiasesSGD(sgd_config),MatrixFactorizationWithBiasesALS(als_config)]
    if SGD:
        return MatrixFactorizationWithBiasesSGD(sgd_config)
    return MatrixFactorizationWithBiasesALS(als_config)
