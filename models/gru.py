# 导入 keras 等相关包
import pandas as pd
import numpy as np
import os, keras
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from data_files import *
import random

#dense_features = ["usr", "used_1", "used_2", "csw", "int"]

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
target = targets

#target = ["CPU_POWER"]  # 或者target = ["DRAM_POWER"]预测内存功耗
time_step = 3
epochs = 3000
gru_units = 128
dense_units = 256

lr = 0.001


class Metrics(Callback):  # 自定义回调函数，在每个epoch结束后进行validate和predict
    def __init__(self, valid_data, test_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_error, test_error = 0, 0
        # val_predict = self.model.predict(self.validation_data[0])

        val_predict = self.model.predict(self.validation_data[0])
        val_targ = self.validation_data[1]

        val_mae = round(mean_absolute_error(val_targ, val_predict), 4)
        logs['val_mae'] = val_mae
        for x, y in zip(val_targ, val_predict):
            x, y = x[0], y[0]
            val_error += abs(float(x) - float(y)) / float(x)
        val_error_rate = val_error / len(val_targ)

        # test_predict = self.model.predict(self.test_data[0])
        test_predict = self.model.predict(self.test_data[0])
        test_targ = self.test_data[1]

        for x, y in zip(test_targ, test_predict):
            x, y = x[0], y[0]
            test_error += abs(float(x) - float(y)) / float(x)
        test_error_rate = test_error / len(test_targ)

        test_mae = round(mean_absolute_error(test_targ, test_predict), 4)
        logs['test_mae'] = test_mae

        print(" \n%f \t%f  \t%f \t%f" % (
        val_mae, val_error_rate, test_mae, test_error_rate))
        return


class Error_rate(Callback):
    def __init__(self, data, label):
        super(Error_rate, self).__init__()
        self.data = data
        self.label = label

    def on_epoch_end(self, epoch, logs=None):
        error = 0
        result = self.model.predict(self.data)
        for x, y in zip(self.label, result):
            x, y = x[0], y[0]
            # print(x,y)
            error += abs(float(x) - float(y)) / float(x)
        error_rate = error / len(self.label)
        logs['test_error_rate'] = error_rate
        print(" — test_error_rate: %f" % (error_rate))


class Mae(Callback):
    def __init__(self, data, label):
        super(Mae, self).__init__()
        self.data = data
        self.label = label

    def on_epoch_end(self, epoch, logs=None):
        error = 0
        result = self.model.predict(self.data)

        for x, y in zip(self.label, result):
            x, y = x[0], y[0]
            # print(x,y)
            error += abs(float(x) - float(y))
        mae = error / len(self.label)
        logs['test_mae'] = mae
        print(" — test_mae: %f" % (mae))


'''这个不管
def get_model():
    def merge(input):
        return K.stack([input[0], input[1],input[2]], 1)
    input1 = Input(shape = (len(dense_features),))
    input2 = Input(shape = (len(dense_features),))
    input3 = Input(shape = (len(dense_features),))

    dense = Dense(len(dense_features),activation = "sigmoid")

    converted1 = dense(input1)
    converted2 = dense(input2)
    converted3 = dense(input3)

    tem = Lambda(merge,name = "merge",output_shape=(time_step,len(dense_features),))([converted1,converted2,converted3])

    ans = GRU(units=128, return_sequences=False, input_shape=(time_step,5,))(tem)

    output = Dense(1)(ans)
    model = Model([input1,input2,input3],output)
    model.compile(loss="mse", optimizer=Adam(0.001))

    print(model.summary())

    return model
'''


def get_gru_model():
    input = Input(shape=(time_step, len(dense_features),))
    ans = GRU(units=128, return_sequences=False, input_shape=(time_step, len(dense_features),))(input)

    output = Dense(1)(ans)
    model = Model(input, output)
    model.compile(  # loss="mse",
        loss=error_rate_loss,
        optimizer=Adam(0.001))

    print(model.summary())

    return model


def get_lstm_model():
    input = Input(shape=(time_step, len(dense_features),))
    ans = LSTM(units=128, return_sequences=False, input_shape=(time_step, len(dense_features),))(input)

    output = Dense(1)(ans)
    model = Model(input, output)
    model.compile(  # loss="mse",
        loss=error_rate_loss,
        optimizer=Adam(lr))

    print(model.summary())

    return model


def error_rate_loss(true, pred):
    a = Lambda(lambda x: K.abs(x[0] - x[1]))([true, pred])
    error_rate = Lambda(lambda x: x[0] / x[1])([a, true])
    return error_rate

#构造GRU的输入
def create_dataset(dataset, label, look_back=time_step):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for p in range(0, data_scale):
        i = random.randint(0, len(dataset) - look_back - 2)
        a = dataset[i:(i + look_back)]
        dataX.append(a.values)
        dataY.append(label[i + look_back - 1])
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
        data, label = create_dataset(data[dense_features], np.array(data[targets]))
        print(len(data))
        result_data = result_data + data
        result_label = result_label + label
    print(len(result_data))
    return np.array(result_data), np.array(result_label)
'''
def get_data(paths, dense_features, target):
    result_data = []
    result_target = []
    for path in paths:
        data = pd.read_csv(path)
        data[dense_features] = data[dense_features].fillna(0, )
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        result_data = result_data + data[dense_features].values.tolist()
        result_target = result_target + data[target].values.tolist()
    #return np.array(data[dense_features]), np.array(data[target])
    return np.array(result_data), np.array(result_target)

def create_dataset(dataset, label, look_back=time_step):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(label[i + look_back - 1])
    return np.array(dataX), np.array(dataY)
'''

def main(train_path, test_path, root):
    if os.path.exists(root) == False:
        os.mkdir(root, mode=0o777)
    data, label = get_data(train_path)
    test, test_label = get_data(test_path)

    #data, label = create_dataset(data, label, time_step)
    #test, test_label = create_dataset(test, test_label, time_step)
    over_all_test_mae = 0
    over_all_test_error_rate = 0
    # 监控验证集准确率并保存每次训练最好模型
    # kf = RepeatedKFold(n_splits=split_num, n_repeats=repeat_num, random_state=2020)
    # for train_index, test_index in kf.split(data):
    model = get_gru_model()
    x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2)
    print(x_val.shape)

    checkpoint = ModelCheckpoint(root + '/final_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    #metrics = Metrics(valid_data=[x_val[:, 0, :], x_val[:, 1, :], x_val[:, 2, :], y_val],
    #                  test_data=[test[:, 0, :], test[:, 1, :], test[:, 2, :], test_label])
    metrics = Metrics(valid_data=[x_val, y_val], test_data=[test,test_label])
    model.fit(x_train, y_train, batch_size=512, epochs=epochs, verbose=0,
              validation_data=[x_val, y_val],
              callbacks=[metrics,
                         checkpoint])

    '''
    model.fit([x_train[:,0,:],x_train[:,1,:],x_train[:,2,:]], y_train,batch_size=32, epochs=epochs, verbose=1,
        validation_data=[[x_val[:,0,:],x_val[:,1,:],x_val[:,2,:]],y_val],
        callbacks=[metrics,
        checkpoint] )   
    '''
    # 预测阶段
    model = get_gru_model()
    model.load_weights(root + '/final_model.h5')
    pred_ans = model.predict(test)
    error = 0
    for x, y in zip(test_label, pred_ans):
        x, y = x[0], y[0]
        error += abs(float(x) - float(y)) / float(x)
    test_error_rate = error / len(pred_ans)

    test_mae = round(mean_absolute_error(test_label, pred_ans), 4)

    print("test_error_rate :", test_error_rate)
    print("test_mae :", test_mae)

    model = get_gru_model()
    model.load_weights(root + '/final_model.h5')
    pred_ans = model.predict(x_val)
    error = 0
    for x, y in zip(y_val, pred_ans):
        x, y = x[0], y[0]
        error += abs(float(x) - float(y)) / float(x)
    val_error_rate = error / len(pred_ans)

    val_mae = round(mean_absolute_error(y_val, pred_ans), 4)

    print("val_error_rate :", val_error_rate)
    print("val_mae :", val_mae)

    f = open(root + '/overall_metrics.txt', "w", encoding="utf8")
    f.write("train_data: " + train_path + "\n")
    f.write("test_data: " + test_path + "\n")
    f.write("epochs: " + str(epochs) + "\n")
    f.write("time_step: " + str(time_step) + "\n")
    f.write("gru_units: " + str(gru_units) + "\n")
    # f.write("dense_units: "+str(dense_units)+"\n")
    f.write("val_mae: " + str(val_mae) + "\n")
    f.write("test_mae: " + str(test_mae) + "\n")
    f.write("val_error_rate: " + str(val_error_rate) + "\n")
    f.write("test_error_rate: " + str(test_error_rate) + "\n")
    f.close()


if __name__ == "__main__":
    main(train_paths, test_paths,model_path)



