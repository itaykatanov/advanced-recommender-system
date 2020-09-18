import skopt

from optimization_objects import Config

# for submission
FIT_ON_TRAIN_VALIDATION = False
# sgd or als
SGD = True
# HYPER_PARAM_SEARCH or manual config
HYPER_PARAM_SEARCH = False
HYPER_PARAM_SEARCH_N_ITER = 10
SEED = 3
MEASURE = 'rmse'

# hyper parameter tuning
SGD_SPACE = [skopt.space.Real(0.005, 0.03, name='lr', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_users', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_items', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_users_bias', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_items_bias', prior='uniform'),
             skopt.space.Categorical([16, 18, 20, 24, 28, 32], name='h_len')]

SGD_CONFIG = Config(
    print_metrics=True,
    beta=0.9,
    hidden_dimension=18,
    lr=0.025,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=25,
    bias_epochs=5,
    seed=SEED)

ALS_SPACE = [skopt.space.Real(0.1, 0.9, name='l2_users', prior='uniform'),
             skopt.space.Real(0.1, 0.9, name='l2_items', prior='uniform'),
             skopt.space.Real(0.1, 0.9, name='l2_users_bias', prior='uniform'),
             skopt.space.Real(0.1, 0.9, name='l2_items_bias', prior='uniform'),
             skopt.space.Categorical([8, 16, 20, 24], name='h_len')]

space = SGD_SPACE if SGD else ALS_SPACE

ALS_CONFIG = Config(
    print_metrics=True,
    hidden_dimension=8,
    l2_users=0.869088,
    l2_items=0.896846,
    l2_users_bias=0.870317,
    l2_items_bias=0.896482,
    epochs=22,
    bias_epochs=2,
    seed=SEED)

# best starting point
if SGD:
    x0 = [[0.025, 0.01, 0.01, 0.001, 0.001, 18]]
    y0 = [0.8945]
else:
    x0 = [[0.892456, 0.897624, 0.110739, 0.565517, 8]]
    y0 = [0.915]

TRAIN_PATH = 'data/Train.csv'
VALIDATION_PATH = 'data/Validation.csv'
TEST_PATH = 'data/Test.csv'
USERS_COL_NAME = 'User_ID_Alias'
ITEMS_COL_NAME = 'Movie_ID_Alias'
RATING_COL_NAME = 'Ratings_Rating'
USER_COL = 'user'
ITEM_COL = 'item'
model_name = 'sgd' if SGD else 'als'
CHECKPOINT_NAME = f"./checkpoint_{model_name}.pkl"
HYPER_PARAM_FILE_NAME = f"HyperParamResult_{model_name}.pkl"
