## 基础工具
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import os

## 读取树模型数据
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
tree_data_path = path+'/user_data/'
Train_data = pd.read_csv(tree_data_path+'train_tree.csv', sep=' ')
TestA_data = pd.read_csv(tree_data_path+'text_tree.csv', sep=' ')

numerical_cols = Train_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price','SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
X_test = TestA_data[feature_cols]
print(X_data.shape)
print(X_test.shape)

X_data = np.array(X_data)
X_test = np.array(X_test)
Y_data = np.array(Train_data['price'])

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
    print("fold n°{}".format(fold_))
    k_x_train = X_data[train_index]
    k_y_train = Y_data[train_index]
    k_x_vali = X_data[vali_index]
    k_y_vali = Y_data[vali_index]
    cb_params = {
        'n_estimators': 1000000, #迭代次数
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
    model_cb = CatBoostRegressor(**cb_params)
    # train the model
    model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=300, early_stopping_rounds=300)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(X_test, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
    predictions_train_cb += model_cb.predict(X_data, ntree_end=model_cb.best_iteration_) / kfolder.n_splits

print("catboost score: {:<8.8f}".format(mean_absolute_error(np.expm1(oof_cb), np.expm1(Y_data))))

output_path = path + '/user_data/'
# 测试集输出
predictions = predictions_cb
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path+'cab_test.csv', index=False)


# 验证集输出
oof_cb[oof_cb < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_data.SaleID
sub['price'] = oof_cb
sub.to_csv(output_path+'cab_train.csv', index=False)

