## 基础工具
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
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
lightgbm
"""
#自定义损失函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_absolute_error(np.expm1(label), np.expm1(preds))
    return 'myFeval', score, False

param = {'boosting_type': 'gbdt',
         'num_leaves': 31,
         'max_depth': -1,
         "lambda_l2": 2,  # 防止过拟合
         'min_data_in_leaf': 20,  # 防止过拟合，好像都不用怎么调
         'objective': 'regression_l1',
         'learning_rate': 0.02,
         "min_child_samples": 20,

         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'mae',
         }
folds = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(X_data))
predictions_lgb = np.zeros(len(X_test))
predictions_train_lgb = np.zeros(len(X_data))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_data[trn_idx], Y_data[trn_idx])
    val_data = lgb.Dataset(X_data[val_idx], Y_data[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=300,
                    early_stopping_rounds=300, feval = myFeval)
    oof_lgb[val_idx] = clf.predict(X_data[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    predictions_train_lgb += clf.predict(X_data, num_iteration=clf.best_iteration) / folds.n_splits

print("lightgbm score: {:<8.8f}".format(mean_absolute_error(np.expm1(oof_lgb), np.expm1(Y_data))))


output_path = path + '/user_data/'
# 测试集输出
predictions = predictions_lgb
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path+'lgb_test.csv', index=False)


# 验证集输出
oof_lgb[oof_lgb < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_data.SaleID
sub['price'] = oof_lgb
sub.to_csv(output_path+'lgb_train.csv', index=False)

