## 基础工具
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os

"""
神经网络
"""
## 读取神经网络模型数据
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
tree_data_path = path+'/user_data/'
Train_NN_data = pd.read_csv(tree_data_path+'train_nn.csv', sep=' ')
Test_NN_data = pd.read_csv(tree_data_path+'test_nn.csv', sep=' ')

numerical_cols = Train_NN_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price','SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_NN_data[feature_cols]
X_test = Test_NN_data[feature_cols]


x = np.array(X_data)
y = np.array(Train_NN_data['price'])
x_test = np.array(X_test)


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


kfolder = KFold(n_splits=5, shuffle=True, random_state=2018)
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
sub.to_csv(output_path+'nn_test.csv', index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_NN_data.SaleID
sub['price'] = oof_nn
sub.to_csv(output_path+'nn_train.csv', index=False)
