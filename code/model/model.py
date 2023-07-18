## 基础工具
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

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
sub.to_csv(output_path+'test_lgb.csv', index=False)


# 验证集输出
oof_lgb[oof_lgb < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = oof_lgb
sub.to_csv(output_path+'train_lgb.csv', index=False)


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
        'n_estimators': 1000000,
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
sub.to_csv(output_path+'test_cab.csv', index=False)


# 验证集输出
oof_cb[oof_cb < 0] = 0
sub['SaleID'] = Train_data.SaleID
sub['price'] = oof_cb
sub.to_csv(output_path+'train_cab.csv', index=False)

"""
神经网络
"""
## 读取神经网络模型数据
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
tree_data_path = path+'/user_data/'
Train_NN_data = pd.read_csv(tree_data_path+'train_nn.csv', sep=' ')
Test_NN_data = pd.read_csv(tree_data_path+'test_nn.csv', sep=' ')

numerical_cols = Train_NN_data.columns
print(numerical_cols)
feature_cols = [col for col in numerical_cols if col not in ['price','SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_NN_data[feature_cols]
X_test = Test_NN_data[feature_cols]
print(X_data.shape)
print(X_test.shape)

x = np.array(X_data)
y = np.array(Train_NN_data['price'])
x_test = np.array(X_test)

print(x)
print(x_test)
print(y)
#调整训练过程的学习率
def scheduler(epoch):
    # 到规定的epoch，学习率减小为原来的1/10

    if epoch  == 1400 :
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    if epoch  == 1700 :
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    if epoch  == 1900 :
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)


kfolder = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_nn = np.zeros(len(x))
predictions_nn = np.zeros(len(x_test))
predictions_train_nn = np.zeros(len(x))
kfold = kfolder.split(x, y)
fold_ = 0
for train_index, vali_index in kfold:
    k_x_train = x[train_index]
    k_y_train = y[train_index]
    k_x_vali = x[vali_index]
    k_y_vali = y[vali_index]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.02)))

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'])

    model.fit(k_x_train,k_y_train,batch_size =512,epochs=2000,validation_data=(k_x_vali, k_y_vali), callbacks=[reduce_lr])#callbacks=callbacks,
    oof_nn[vali_index] = model.predict(k_x_vali).reshape((model.predict(k_x_vali).shape[0],))
    predictions_nn += model.predict(x_test).reshape((model.predict(x_test).shape[0],)) / kfolder.n_splits
    predictions_train_nn += model.predict(x).reshape((model.predict(x).shape[0],)) / kfolder.n_splits

print("NN score: {:<8.8f}".format(mean_absolute_error(oof_nn, y)))

output_path = path + '/user_data/'
# 测试集输出
predictions = predictions_nn
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Test_NN_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path+'test_nn.csv', index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_NN_data.SaleID
sub['price'] = oof_nn
sub.to_csv(output_path+'train_nn.csv', index=False)


tree_data_path = path+'/user_data/'

#导入树模型lgb预测数据，进行二层stacking输出
predictions_lgb = np.array(pd.read_csv(tree_data_path+'lgb_test.csv')['price'])
oof_lgb = np.array(pd.read_csv(tree_data_path+'lgb_train.csv')['price'])

#导入树模型cab预测数据，进行二层stacking输出
predictions_cb = np.array(pd.read_csv(tree_data_path+'cab_test.csv')['price'])
oof_cb = np.array(pd.read_csv(tree_data_path+'cab_train.csv')['price'])

#读取price，对验证集进行评估
Train_data = pd.read_csv(tree_data_path+'train_tree.csv', sep=' ')
TestA_data = pd.read_csv(tree_data_path+'text_tree.csv', sep=' ')
Y_data = Train_data['price']

train_stack = np.vstack([oof_lgb, oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_cb]).transpose()
folds_stack = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2018)
tree_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

#二层贝叶斯回归stack
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, Y_data)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], Y_data[trn_idx]
    val_data, val_y = train_stack[val_idx], Y_data[val_idx]

    Bayes = linear_model.BayesianRidge()
    Bayes.fit(trn_data, trn_y)
    tree_stack[val_idx] = Bayes.predict(val_data)
    predictions += Bayes.predict(test_stack) / 20

tree_predictions = np.expm1(predictions)
tree_stack = np.expm1(tree_stack)
tree_point = mean_absolute_error(tree_stack, np.expm1(Y_data))
print("树模型：二层贝叶斯: {:<8.8f}".format(tree_point))




#导入神经网络模型预测训练集数据，进行三层融合
predictions_nn = np.array(pd.read_csv(tree_data_path+'nn_test.csv')['price'])
oof_nn = np.array(pd.read_csv(tree_data_path+'nn_train.csv')['price'])

nn_point = mean_absolute_error(oof_nn, np.expm1(Y_data))
print("神经网络: {:<8.8f}".format(nn_point))

oof = (oof_nn + tree_stack)/2
predictions = (tree_predictions + predictions_nn)/2
all_point = mean_absolute_error(oof, np.expm1(Y_data))
print("总输出：三层融合: {:<8.8f}".format(all_point))


output_path = path + '/prediction_result/'
# 测试集输出
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
predictions[predictions < 0] = 0
sub['price']=predictions
sub.to_csv(output_path+'predictions.csv', index=False)