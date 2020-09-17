import pandas as pd
import keras
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.callbacks import Callback
from data_files import *
import random

train_paths = []
test_paths = []

train_datas = ['PARSEC', 'hpcc', 'hpl-s', 'hpcg', 'SPEC', 'GRAPH500', 'HPL-AI', 'LmBench','MiBench','RoyBench','SMG2000']
test_datas = []
test_No = 10
train_datas = train_datas[:test_No]+train_datas[test_No+1:]
test_datas = train_datas[test_No:test_No+1]
model_path = "./models/"#用于保存模型文件
if os.path.exists(model_path)==False:#帮助构造model保存路径
    os.mkdir(model_path, mode=0o777)

for data_file in data_files:
    if 'HPCC' in data_file:
        if not 'single' in data_file:
            continue
    if 'SPEC' in data_file:
        if not 'ref' in data_file:
            continue
    if 'PARSEC' in data_file:
        if not 'simlarge' in data_file:
            continue
    isTrainData = False
    for label in train_datas:
        if label in data_file:
            isTrainData = True
            break
    if isTrainData:
        train_paths.append(data_file)
    else:
        test_paths.append(data_file)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
cpu_features = ['cpu-cycles', 'L1-dcache-load-misses', 'dTLB-loads', 'L1-dcache-loads', 'bus-cycles', 'ref-cycles', 'instructions', 'LLC-prefetches', 'L1-dcache-prefetch-misses', 'branch-loads']
mem_features = ['uncore_imc_0/cas_count_read/', 'uncore_imc_1/cas_count_read/', 'uncore_imc_2/cas_count_read/', 'uncore_imc_3/cas_count_read/', 'LLC-prefetch-misses',
                'node-prefetches', 'node-loads', 'LLC-load-misses', 'LLC-loads', 'cache-misses']
dense_features = mem_features
data_scale = 10000
#targets = ["power/energy-pkg/"] #target用于定义被预测的目标是什么，可以选择切换为"power/energy-cores/" 或者 下一行代码中的 "power/energy-ram/"
targets = ["power/energy-ram/"]
lr = 0.001
class Error_rate(Callback):
    def __init__(self, val_data, val_label,test_data,test_label,target):
        super(Error_rate, self).__init__()
        self.val_data = val_data
        self.val_label = val_label
        self.test_data = test_data
        self.test_label = test_label
        self.target = target
        self.error_rate = 1
    def on_epoch_end(self, epoch, logs=None):
        error = 0
        result = self.model.predict(self.val_data)
        for x, y in zip(self.val_label, result):
            error += abs(x - float(y)) / float(x)
        error_rate = error / len(self.val_label)
        logs[self.target +'val mean_absolute_error'] = mean_absolute_error(self.val_label, result)
        print(self.target+ "val mean_absolute_error: %f" % (mean_absolute_error(self.val_label, result)))
        logs[self.target+' val_error_rate'] = error_rate
        print(self.target +" val_error_rate: %f" % (error_rate))
        if error_rate <self.error_rate:#只有当验证集上的Eerror_rate降低以后，才对测试集进行测试，进而保存模型文件
            self.error_rate = error_rate
            self.model.save(model_path+"/error_rate_"+str(error_rate)+"_.h5")
            error = 0
            result = self.model.predict(self.test_data)
            for x, y in zip(self.test_label, result):
                error += abs(x - float(y)) / float(x)
            test_error_rate = error / len(self.test_label)
            logs[self.target +'test mean_absolute_error'] = mean_absolute_error(self.test_label, result)
            print(self.target+ "test mean_absolute_error: %f" % (mean_absolute_error(self.test_label, result)))
            logs[self.target+' test_error_rate'] = test_error_rate
            print(self.target +" test_error_rate: %f" % (test_error_rate))

#class Error_rate(Callback):
#    def __init__(self, data,label):
#        super(Error_rate, self).__init__()
#        self.data = data
#        self.label = label
#    def on_epoch_end(self, epoch, logs=None):
#        error = 0
#        result = self.model.predict(self.data)
#        for x,y in zip(self.label,result):
#            #print(x,y)
#            x,y = x[0],y[0]
#            print(x,y)
#            error+=abs(float(x)-float(y))/float(x)
#        error_rate = error/len(self.label)
#        logs['train_error_rate'] = error_rate
#        print(" — train_error_rate: %f" % (error_rate))
def get_model(data_shape):
    def error_rate_loss(true, pred):
        a = Lambda(lambda x: K.abs(x[0] - x[1]))([true, pred])
        error_rate = Lambda(lambda x: x[0] / x[1])([a, true])
        return error_rate
    model = keras.Sequential([
    keras.layers.Dense(256, activation="relu",
                       input_shape=(data_shape,)),
    #keras.layers.Dense(256, activation="relu"),
    #keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
  ])

    model.compile(optimizer=Adam(lr),
                    #loss="mae",
                    loss = error_rate_loss)#loss也可以使用直接优化MRE
    print(model.summary())
    return model
def create_dataset(dataset, label):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for p in range(0, data_scale):
        i = random.randint(0, len(dataset) - 2)
        dataX.append(dataset.values[i])
        dataY.append(label[i])
    return dataX, dataY

def get_data(paths):
    result_data = []
    result_label = []
    print(paths)
    for path in paths:
        print(path)
        data = pd.read_csv(open(path))
        data["target"] = pd.Series(np.zeros(data["power/energy-pkg/"].shape[0]))
        data[dense_features] = data[dense_features].fillna(0,)
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        special_node = ["target"]
        data, label = create_dataset(data[special_node + dense_features], np.array(data[targets]))
        print(len(data))
        result_data = result_data + data
        result_label = result_label + label
    print(len(result_data))
    return np.array(result_data), np.array(result_label)

if __name__ == "__main__":
    train_data, train_label = get_data(train_paths)#加载训练集
    test_data, test_label = get_data(test_paths)#加载测试集
    
    #data[dense_features] = data[dense_features].fillna(0,)

    #mms = MinMaxScaler(feature_range=(0,1))
    #data[dense_features] = mms.fit_transform(data[dense_features])
    model = linear_model.LinearRegression()
    #model = get_model(11)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=2020)

    error_rate = Error_rate(X_val, y_val, test_data, test_label, targets[0])

    model.fit(X_train,y_train)
    pred_ans = model.predict(test_data)
    error = 0
    for x, y in zip(test_label, pred_ans):
        x, y = x[0], y[0]
        error += abs(float(x) - float(y)) / float(x)
    test_error_rate = error / len(pred_ans)  # 计算error_rate

    test_mae = round(mean_absolute_error(test_label, pred_ans), 6)

    print("test_error_rate :", round(test_error_rate, 6))
    print("test_mae :", test_mae)
    #pred_ans = model.predict(test[dense_features+sparse_features], batch_size=256)
