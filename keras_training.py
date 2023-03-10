import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix # will plot the confusion matrix
import time

import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('UNSW_NB15_training-set.csv')
df.describe(include='all')

list_drop = ['id','attack_cat']
df.drop(list_drop,axis=1,inplace=True) # 删去数据中的index列和攻击类型列

# Clamp extreme Values 去除异常值
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')

DEBUG =0 # 0则不输出处理信息时每一列的关键数据，1则输出相关数据

for feature in df_numeric.columns: # 遍历所有数据类型为数字的列，进行修改
    if DEBUG == 1:
        print(feature)
        print('max = '+str(df_numeric[feature].max()))
        print('75th = '+str(df_numeric[feature].quantile(0.95)))
        print('median = '+str(df_numeric[feature].median()))
        print(df_numeric[feature].max()>10*df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
        df[feature] = np.where(df[feature]<df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

# Apply log function to nearly all numeric, since they are all mostly skewed to the right
df_numeric = df.select_dtypes(include=[np.number])
df_before = df_numeric.copy()

for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique()>50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique()>50:
        if df_numeric[feature].min()==0:
            df[feature] = np.log(df[feature]+1)
        else:
            df[feature] = np.log(df[feature])

df_numeric = df.select_dtypes(include=[np.number])

# Reduce the labels in catagorical features
df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')

for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique() > 6)
        print(sum(df[feature].isin(df[feature].value_counts().head().index)))
        print('----------------------------------------------------')

    if df_cat[feature].nunique() > 6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')

df['proto'].value_counts().head().index

df['proto'].value_counts().index

# Encode categorical features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X.head()
feature_names = list(X.columns)
np.shape(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
np.shape(X)
df_cat.describe(include='all')

for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

# Split test and training

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify=y)
df_cat.describe(include='all')

# 6 + 5 + 6 unique = 17, therefore the first 17 rows will be the categories that have been encoded, start scaling from row 18 only.
sc = StandardScaler()
X_train[:, 18:] = sc.fit_transform(X_train[:, 18:])
X_test[:, 18:] = sc.transform(X_test[:, 18:])

model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','time to train','time to predict','total time'])

# GRU(keras)
#Build the neural network model

from tensorflow.keras.layers import BatchNormalization, Dropout

def build_model(drop):
    model = Sequential()

    model.add(BatchNormalization())
    model.add(GRU(128, input_shape=(1, 56), return_sequences=True))  # 添加了一层GRU模型，设定其输入尺寸（为get_GRU函数的第一个参数）为shape
    model.add(Dropout(drop))  # Dropout以防止模型过拟合

    model.add(BatchNormalization())
    model.add(GRU(128, return_sequences=True))  # 此处就无需设定输入尺寸，只有第一层需要设定，后续均会自动计算
    model.add(Dropout(drop))

    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))  # Dense层用于维度变换
    model.add(Dropout(drop))

    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))  # Dense层用于维度变换
    model.add(Dropout(drop))

    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))  # Dense层又名全连接层，用于对向量进行非线性变换，映射到其他的向量空间
    opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)  # Adam是一个优化器（自适应矩估计），根据学习进程自动调整学习目标
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.summary()
    
    return model

#The GRU input layer must be 3D.
#The meaning of the 3 input dimensions are: samples, time steps, and features.
#reshape input data
X_train_array = array(X_train) #array has been declared in the previous cell
print(len(X_train_array))
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0],1,56)

#reshape output data
X_test_array=  array(X_test)
X_test_reshaped = X_test_array.reshape(X_test_array.shape[0],1,56)


#institate the model
model = build_model(0.30)

start = time.time()
#fit the model
print("GRU_model")
model.fit(X_train_reshaped, y_train, epochs=200, batch_size=2000,verbose=2)
end_train = time.time()

loss, accuracy = model.evaluate(X_test_reshaped, y_test)
# loss, accuracy, f1s, precision, recall = model.evaluate(X_test_reshaped, y_test)
end_predict = time.time()
model_performance.loc['GRU (Keras)'] = [accuracy, accuracy, accuracy, accuracy, end_train-start,end_predict-end_train,end_predict-start]

# 尝试optuna优化超参数
def objective(trial):
    x = trial.suggest_uniform('x',0,1)
    model = build_model(x)
    start = time.time()
    print("auto-try")
    model.fit(X_train_reshaped, y_train, epochs=200, batch_size=2000,verbose=2)
    end_train = time.time()
    return model.loss

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)


# 下面开始构建自己的模型，可以尝试复现原论文的FastGRU或者FastLSTM


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
