## 基础工具
import warnings
warnings.filterwarnings('ignore')
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

## 读取树模型数据
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
tree_data_path = path + '/user_data/'

try:
    train_file = tree_data_path + 'train_tree.csv'
    test_file = tree_data_path + 'text_tree.csv'
    logger.info("Loading training data from: %s", train_file)
    logger.info("Loading test data from: %s", test_file)

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    Train_data = pd.read_csv(train_file, sep=' ')
    TestA_data = pd.read_csv(test_file, sep=' ')
    logger.info("Training data loaded successfully: %d rows, %d columns", Train_data.shape[0], Train_data.shape[1])
    logger.info("Test data loaded successfully: %d rows, %d columns", TestA_data.shape[0], TestA_data.shape[1])
except FileNotFoundError as e:
    logger.error("Data file not found: %s", e)
    sys.exit(1)
except pd.errors.EmptyDataError:
    logger.error("Data file is empty or cannot be parsed.")
    sys.exit(1)
except Exception as e:
    logger.error("Unexpected error loading data: %s", e)
    sys.exit(1)

# Validate required columns
if 'price' not in Train_data.columns:
    logger.error("'price' column missing from training data.")
    sys.exit(1)
if 'SaleID' not in Train_data.columns or 'SaleID' not in TestA_data.columns:
    logger.error("'SaleID' column missing from data.")
    sys.exit(1)

numerical_cols = Train_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price', 'SaleID']]

if len(feature_cols) == 0:
    logger.error("No feature columns found after excluding 'price' and 'SaleID'.")
    sys.exit(1)

logger.info("Using %d feature columns.", len(feature_cols))

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
X_test = TestA_data[feature_cols]
logger.info("X_data shape: %s", X_data.shape)
logger.info("X_test shape: %s", X_test.shape)

X_data = np.array(X_data)
X_test = np.array(X_test)
Y_data = np.array(Train_data['price'])

# Check for NaN/Inf in data
if np.isnan(X_data).any() or np.isinf(X_data).any():
    logger.warning("Training features contain NaN or Inf values. This may affect model performance.")
if np.isnan(X_test).any() or np.isinf(X_test).any():
    logger.warning("Test features contain NaN or Inf values. This may affect model performance.")
if np.isnan(Y_data).any() or np.isinf(Y_data).any():
    logger.warning("Training labels contain NaN or Inf values. This may affect model performance.")

"""
catboost
"""
kfolder = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_cb = np.zeros(len(X_data))
predictions_cb = np.zeros(len(X_test))
predictions_train_cb = np.zeros(len(X_data))
kfold = kfolder.split(X_data, Y_data)
fold_ = 0

for train_index, vali_index in kfold:
    fold_ = fold_ + 1
    logger.info("Training fold n°%d", fold_)
    k_x_train = X_data[train_index]
    k_y_train = Y_data[train_index]
    k_x_vali = X_data[vali_index]
    k_y_vali = Y_data[vali_index]
    cb_params = {
        'n_estimators': 1000000,  # 迭代次数
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.02,
        'depth': 6,
        'use_best_model': True,
        'subsample': 0.6,
        'bootstrap_type': 'Bernoulli',
        'reg_lambda': 3,
        'one_hot_max_size': 2,
    }

    try:
        model_cb = CatBoostRegressor(**cb_params)
        # train the model
        model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=300, early_stopping_rounds=300)
    except Exception as e:
        logger.error("Error training CatBoost model at fold %d: %s", fold_, e)
        raise

    try:
        oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
        predictions_cb += model_cb.predict(X_test, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
        predictions_train_cb += model_cb.predict(X_data, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
        logger.info("Fold %d prediction completed successfully.", fold_)
    except Exception as e:
        logger.error("Error during prediction at fold %d: %s", fold_, e)
        raise

score = mean_absolute_error(np.expm1(oof_cb), np.expm1(Y_data))
logger.info("catboost score: %8.8f", score)

output_path = path + '/user_data/'
if not os.path.exists(output_path):
    logger.warning("Output directory does not exist, creating: %s", output_path)
    os.makedirs(output_path, exist_ok=True)

# 测试集输出
try:
    predictions = predictions_cb
    predictions[predictions < 0] = 0
    sub = pd.DataFrame()
    sub['SaleID'] = TestA_data.SaleID
    sub['price'] = predictions
    output_file = output_path + 'cab_test.csv'
    sub.to_csv(output_file, index=False)
    logger.info("Test predictions saved to: %s", output_file)
except Exception as e:
    logger.error("Error saving test predictions: %s", e)
    raise

# 验证集输出
try:
    oof_cb[oof_cb < 0] = 0
    sub = pd.DataFrame()
    sub['SaleID'] = Train_data.SaleID
    sub['price'] = oof_cb
    output_file = output_path + 'cab_train.csv'
    sub.to_csv(output_file, index=False)
    logger.info("Train (OOF) predictions saved to: %s", output_file)
except Exception as e:
    logger.error("Error saving train predictions: %s", e)
    raise

logger.info("CatBoost model pipeline completed successfully.")

