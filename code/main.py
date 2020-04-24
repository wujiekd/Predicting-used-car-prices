#导入需要使用的库
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
input_path = path + '/data/'
Train_data = pd.read_csv(input_path+'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(input_path+'used_car_testA_20200313.csv', sep=' ')


"""
—————————————————————————————————————————————以下为树模型的数据处理—————————————————————————————————————————————
"""
"""
一、预测值处理，处理目标值长尾分布的问题
"""
Train_data['price'] = np.log1p(Train_data['price'])

# 合并方便后面的操作
df = pd.concat([Train_data, Test_data], ignore_index=True)

"""
二、数据简单预处理，分三步进行
"""
## 1、第一步处理无用值和基本无变化的值
#SaleID肯定没用，但是我们可以用来统计别的特征的group数量
#name一般没什么好挖掘的，不过同名的好像不少，可以挖掘一下
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
del df['name']


#seller有一个特殊值，训练集特有测试集没有，把它删除掉
df.drop(df[df['seller'] == 1].index, inplace=True)
del df['offerType']
del df['seller']


## 2、第二步处理缺失值
# 以下特征全部填充众数
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)



## 3、第三步处理异常值

# 异常值就目前初步判断，只有notRepairedDamage的值有问题，还有题目规定了范围的power。处理一下
df['power'] = df['power'].map(lambda x: 600 if x>600 else x)
df['notRepairedDamage'] = df['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype('float32')

"""
三、以上为数据简单预处理，以下为特征工程（特征工程搞起来，分三大块整理一下）
"""
## 1、时间，地区啥的

#时间
from datetime import datetime
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date

df['regDate'] = df['regDate'].apply(date_process)
df['creatDate'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)


#地区
df['regionCode_count'] = df.groupby(['regionCode'])['SaleID'].transform('count')
df['city'] = df['regionCode'].apply(lambda x : str(x)[:2])


## 2、分类特征
# 对可分类的连续特征进行分桶，kilometer是已经分桶了
bin = [i*10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)
tong = df[['power_bin', 'power']].head()


bin = [i*10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)
tong = df[['model_bin', 'model']].head()

# 将稍微取值多一点的分类特征与price进行特征组合，做了非常多组，但是在最终使用的时候，每组分开测试，挑选真正work的特征
Train_gb = Train_data.groupby("regionCode")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['regionCode_amount'] = len(kind_data)
    info['regionCode_price_max'] = kind_data.price.max()
    info['regionCode_price_median'] = kind_data.price.median()
    info['regionCode_price_min'] = kind_data.price.min()
    info['regionCode_price_sum'] = kind_data.price.sum()
    info['regionCode_price_std'] = kind_data.price.std()
    info['regionCode_price_mean'] = kind_data.price.mean()
    info['regionCode_price_skew'] = kind_data.price.skew()
    info['regionCode_price_kurt'] = kind_data.price.kurt()
    info['regionCode_mad'] = kind_data.price.mad()

    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "regionCode"})
df = df.merge(brand_fe, how='left', on='regionCode')

Train_gb = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_mean'] = kind_data.price.mean()
    info['brand_price_skew'] = kind_data.price.skew()
    info['brand_price_kurt'] = kind_data.price.kurt()
    info['brand_price_mad'] = kind_data.price.mad()

    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
df = df.merge(brand_fe, how='left', on='brand')

Train_gb = Train_data.groupby("model")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['model_amount'] = len(kind_data)
    info['model_price_max'] = kind_data.price.max()
    info['model_price_median'] = kind_data.price.median()
    info['model_price_min'] = kind_data.price.min()
    info['model_price_sum'] = kind_data.price.sum()
    info['model_price_std'] = kind_data.price.std()
    info['model_price_mean'] = kind_data.price.mean()
    info['model_price_skew'] = kind_data.price.skew()
    info['model_price_kurt'] = kind_data.price.kurt()
    info['model_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "model"})
df = df.merge(brand_fe, how='left', on='model')

Train_gb = Train_data.groupby("kilometer")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['kilometer_amount'] = len(kind_data)
    info['kilometer_price_max'] = kind_data.price.max()
    info['kilometer_price_median'] = kind_data.price.median()
    info['kilometer_price_min'] = kind_data.price.min()
    info['kilometer_price_sum'] = kind_data.price.sum()
    info['kilometer_price_std'] = kind_data.price.std()
    info['kilometer_price_mean'] = kind_data.price.mean()
    info['kilometer_price_skew'] = kind_data.price.skew()
    info['kilometer_price_kurt'] = kind_data.price.kurt()
    info['kilometer_price_mad'] = kind_data.price.mad()

    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "kilometer"})
df = df.merge(brand_fe, how='left', on='kilometer')

Train_gb = Train_data.groupby("bodyType")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['bodyType_amount'] = len(kind_data)
    info['bodyType_price_max'] = kind_data.price.max()
    info['bodyType_price_median'] = kind_data.price.median()
    info['bodyType_price_min'] = kind_data.price.min()
    info['bodyType_price_sum'] = kind_data.price.sum()
    info['bodyType_price_std'] = kind_data.price.std()
    info['bodyType_price_mean'] = kind_data.price.mean()
    info['bodyType_price_skew'] = kind_data.price.skew()
    info['bodyType_price_kurt'] = kind_data.price.kurt()
    info['bodyType_price_mad'] = kind_data.price.mad()

    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "bodyType"})
df = df.merge(brand_fe, how='left', on='bodyType')


Train_gb = Train_data.groupby("fuelType")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['fuelType_amount'] = len(kind_data)
    info['fuelType_price_max'] = kind_data.price.max()
    info['fuelType_price_median'] = kind_data.price.median()
    info['fuelType_price_min'] = kind_data.price.min()
    info['fuelType_price_sum'] = kind_data.price.sum()
    info['fuelType_price_std'] = kind_data.price.std()
    info['fuelType_price_mean'] = kind_data.price.mean()
    info['fuelType_price_skew'] = kind_data.price.skew()
    info['fuelType_price_kurt'] = kind_data.price.kurt()
    info['fuelType_price_mad'] = kind_data.price.mad()

    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "fuelType"})
df = df.merge(brand_fe, how='left', on='fuelType')


# 测试分类特征与price时，发现有点效果，立马对model进行处理
kk = "regionCode"
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['car_age_day'] > 0]
    info[kk+'_days_max'] = kind_data.car_age_day.max()
    info[kk+'_days_min'] = kind_data.car_age_day.min()
    info[kk+'_days_std'] = kind_data.car_age_day.std()
    info[kk+'_days_mean'] = kind_data.car_age_day.mean()
    info[kk+'_days_median'] = kind_data.car_age_day.median()
    info[kk+'_days_sum'] = kind_data.car_age_day.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)

Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['power'] > 0]
    info[kk+'_power_max'] = kind_data.power.max()
    info[kk+'_power_min'] = kind_data.power.min()
    info[kk+'_power_std'] = kind_data.power.std()
    info[kk+'_power_mean'] = kind_data.power.mean()
    info[kk+'_power_median'] = kind_data.power.median()
    info[kk+'_power_sum'] = kind_data.power.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)

## 3、连续数值特征
# 都是匿名特征 比较训练集和测试集分布 分析完 基本没什么问题 先暂且全部保留咯
# 后期也许得对相似度较大的进行剔除处理
# 对简易lgb模型输出的特征重要度较高的几个连续数值特征对price进行刻画

dd = 'v_3'
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data[dd] > -10000000]
    info[kk+'_'+dd+'_max'] = kind_data.v_3.max()
    info[kk+'_'+dd+'_min'] = kind_data.v_3.min()
    info[kk+'_'+dd+'_std'] = kind_data.v_3.std()
    info[kk+'_'+dd+'_mean'] = kind_data.v_3.mean()
    info[kk+'_'+dd+'_median'] = kind_data.v_3.median()
    info[kk+'_'+dd+'_sum'] = kind_data.v_3.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)



dd = 'v_0'
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data[dd]> -10000000]
    info[kk+'_'+dd+'_max'] = kind_data.v_0.max()
    info[kk+'_'+dd+'_min'] = kind_data.v_0.min()
    info[kk+'_'+dd+'_std'] = kind_data.v_0.std()
    info[kk+'_'+dd+'_mean'] = kind_data.v_0.mean()
    info[kk+'_'+dd+'_median'] = kind_data.v_0.median()
    info[kk+'_'+dd+'_sum'] = kind_data.v_0.sum()
    all_info[kind] = info
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)

"""
四、补充的特征工程
"""
## 主要是对匿名特征和几个重要度较高的分类特征进行特征交叉
#第一批特征工程
for i in range(15):
    for j in range(15):
        df['new'+str(i)+'*'+str(j)]=df['v_'+str(i)]*df['v_'+str(j)]


#第二批特征工程
for i in range(15):
    for j in range(15):
        df['new'+str(i)+'+'+str(j)]=df['v_'+str(i)]+df['v_'+str(j)]

# 第三批特征工程
for i in range(15):
    df['new' + str(i) + '*power'] = df['v_' + str(i)] * df['power']

for i in range(15):
    df['new' + str(i) + '*day'] = df['v_' + str(i)] * df['car_age_day']

for i in range(15):
    df['new' + str(i) + '*year'] = df['v_' + str(i)] * df['car_age_year']


#第四批特征工程
for i in range(15):
    for j in range(15):
        df['new'+str(i)+'-'+str(j)]=df['v_'+str(i)]-df['v_'+str(j)]


"""
五、筛选特征
"""
numerical_cols = df.select_dtypes(exclude='object').columns

list_tree = [ 'model_power_sum','price','SaleID',
 'model_power_std', 'model_power_median', 'model_power_max',
 'brand_price_max', 'brand_price_median',
 'brand_price_sum', 'brand_price_std',
 'model_days_sum',
 'model_days_std', 'model_days_median', 'model_days_max', 'model_bin', 'model_amount',
 'model_price_max', 'model_price_median',
 'model_price_min', 'model_price_sum', 'model_price_std',
 'model_price_mean', 'bodyType', 'model', 'brand', 'fuelType', 'gearbox', 'power', 'kilometer',
 'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
 'v_11', 'v_12', 'v_13', 'v_14', 'name_count', 'regDate_year', 'car_age_day', 'car_age_year',
 'power_bin','fuelType', 'gearbox', 'kilometer', 'notRepairedDamage',  'name_count', 'car_age_day', 'new3*3', 'new12*14', 'new2*14','new14*14']

for i in range(15):
    for j in range(15):
        list_tree.append('new'+str(i)+'+'+str(j))

feature_cols = [col for col in numerical_cols if
             col  in
             list_tree]

feature_cols = [col for col in feature_cols if
             col  not in
             ['new14+6', 'new13+6', 'new0+12', 'new9+11', 'v_3', 'new11+10', 'new10+14', 'new12+4', 'new3+4', 'new11+11', 'new13+3', 'new8+1', 'new1+7', 'new11+14', 'new8+13', 'v_8', 'v_0', 'new3+5', 'new2+9', 'new9+2', 'new0+11', 'new13+7', 'new8+11', 'new5+12', 'new10+10', 'new13+8', 'new11+13', 'new7+9', 'v_1', 'new7+4', 'new13+4', 'v_7', 'new5+6', 'new7+3', 'new9+10', 'new11+12', 'new0+5', 'new4+13', 'new8+0', 'new0+7', 'new12+8', 'new10+8', 'new13+14', 'new5+7', 'new2+7', 'v_4', 'v_10', 'new4+8', 'new8+14', 'new5+9', 'new9+13', 'new2+12', 'new5+8', 'new3+12', 'new0+10', 'new9+0', 'new1+11', 'new8+4', 'new11+8', 'new1+1', 'new10+5', 'new8+2', 'new6+1', 'new2+1', 'new1+12', 'new2+5', 'new0+14', 'new4+7', 'new14+9', 'new0+2', 'new4+1', 'new7+11', 'new13+10', 'new6+3', 'new1+10', 'v_9', 'new3+6', 'new12+1', 'new9+3', 'new4+5', 'new12+9', 'new3+8', 'new0+8', 'new1+8', 'new1+6', 'new10+9', 'new5+4', 'new13+1', 'new3+7', 'new6+4', 'new6+7', 'new13+0', 'new1+14', 'new3+11', 'new6+8', 'new0+9', 'new2+14', 'new6+2', 'new12+12', 'new7+12', 'new12+6', 'new12+14', 'new4+10', 'new2+4', 'new6+0', 'new3+9', 'new2+8', 'new6+11', 'new3+10', 'new7+0', 'v_11', 'new1+3', 'new8+3', 'new12+13', 'new1+9', 'new10+13', 'new5+10', 'new2+2', 'new6+9', 'new7+10', 'new0+0', 'new11+7', 'new2+13', 'new11+1', 'new5+11', 'new4+6', 'new12+2', 'new4+4', 'new6+14', 'new0+1', 'new4+14', 'v_5', 'new4+11', 'v_6', 'new0+4', 'new1+5', 'new3+14', 'new2+10', 'new9+4', 'new2+6', 'new14+14', 'new11+6', 'new9+1', 'new3+13', 'new13+13', 'new10+6', 'new2+3', 'new2+11', 'new1+4', 'v_2', 'new5+13', 'new4+2', 'new0+6', 'new7+13', 'new8+9', 'new9+12', 'new0+13', 'new10+12', 'new5+14', 'new6+10', 'new10+7', 'v_13', 'new5+2', 'new6+13', 'new9+14', 'new13+9', 'new14+7', 'new8+12', 'new3+3', 'new6+12', 'v_12', 'new14+4', 'new11+9', 'new12+7', 'new4+9', 'new4+12', 'new1+13', 'new0+3', 'new8+10', 'new13+11', 'new7+8', 'new7+14', 'v_14', 'new10+11', 'new14+8', 'new1+2']]

df = df[feature_cols]


"""
六、导出数据
"""
## 切割数据,导出数据,作为树模型的训练数据

output_path = path + '/user_data/'
tree_data = df
print(tree_data.shape)
train_num = df.shape[0]-50000
tree_data[0:int(train_num)].to_csv(output_path+'train_tree.csv', index=0,sep=' ')
tree_data[train_num:train_num+50000].to_csv(output_path+'text_tree.csv', index=0,sep=' ')

print('树模型数据已经准备完毕~~~~~~~')



"""
—————————————————————————————————————————————以下为神经网络的数据处理—————————————————————————————————————————————
"""
input_path = path + '/data/'
Train_data = pd.read_csv(input_path+'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(input_path+'used_car_testA_20200313.csv', sep=' ')

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
         'learning_rate': 0.01,
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

    num_round = 100000000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=300,
                    early_stopping_rounds=600, feval = myFeval)
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
        'n_estimators': 100000000,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.01,
        'depth': 6,
        'use_best_model': True,
        'subsample': 0.6,
        'bootstrap_type': 'Bernoulli',
        'reg_lambda': 3,
        'one_hot_max_size': 2,
    }
    model_cb = CatBoostRegressor(**cb_params)
    # train the model
    model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=300, early_stopping_rounds=600)
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
sub.to_csv(output_path+'nn_test.csv', index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_NN_data.SaleID
sub['price'] = oof_nn
sub.to_csv(output_path+'nn_train.csv', index=False)


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