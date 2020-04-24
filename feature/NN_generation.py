#导入需要使用的库
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os

path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
input_path = path + '/data/'
Train_data = pd.read_csv(input_path+'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(input_path+'used_car_testA_20200313.csv', sep=' ')
"""
—————————————————————————————————————————————以下为神经网络的数据处理—————————————————————————————————————————————
"""

# 合并方便后面的操作
df = pd.concat([Train_data, Test_data], ignore_index=True)


#选择需要使用的特征标签，由于nn会生成大量的特征，我们只需要保留原始特征和刻画几个明显特征即可
feature = ['model','brand','bodyType','fuelType','kilometer','notRepairedDamage','power','regDate_month','creatDate_year','creatDate_month'
    ,'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
       'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14','car_age_day','car_age_year','regDate_year','name_count']


#处理异常数据
df.drop(df[df['seller'] == 1].index, inplace=True)
#记录一下df的price
df_copy = df

df['power'][df['power']>600]=600

#notRepairedDamage的值是0和1，然后为-的值设置为0.5，在将它进行标签转换，0->1;0.5->2;1->3;这样符合神经网络的特征提取，不确定值位于两个确定值的中间～
df.replace(to_replace = '-', value = 0.5, inplace = True)
le = LabelEncoder()
df['notRepairedDamage'] = le.fit_transform(df['notRepairedDamage'].astype(str))

#日期处理
from datetime import datetime
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date

df['regDates'] = df['regDate'].apply(date_process)
df['creatDates'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDates'].dt.year
df['regDate_month'] = df['regDates'].dt.month
df['regDate_day'] = df['regDates'].dt.day
df['creatDate_year'] = df['creatDates'].dt.year
df['creatDate_month'] = df['creatDates'].dt.month
df['creatDate_day'] = df['creatDates'].dt.day
df['car_age_day'] = (df['creatDates'] - df['regDates']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)

#对name进行挖掘
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')

#填充众数
df.fillna(df.median(),inplace= True)



#特征归一化
scaler = MinMaxScaler()
scaler.fit(df[feature].values)
df= scaler.transform(df[feature].values)


## 切割数据,导出数据,作为神经网络的训练数据
output_path = path + '/user_data/'
nn_data = pd.DataFrame(df,columns=feature)
nn_data['price']=np.array(df_copy['price'])
nn_data['SaleID']=np.array(df_copy['SaleID'])
print(nn_data.shape)
train_num = df.shape[0]-50000
nn_data[0:int(train_num)].to_csv(output_path+'train_nn.csv', index=0, sep=' ')
nn_data[train_num:train_num+50000].to_csv(output_path+'test_nn.csv', index=0, sep=' ')

print('NN模型数据已经准备完毕~~~~~~~')
