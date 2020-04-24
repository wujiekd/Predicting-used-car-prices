## 基础工具
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
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

