import pandas as pd
import keras
import keras.backend as K
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn.model_selection import train_test_split
from keras_gcn import GraphConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import  Model
from keras.layers import *
from keras.callbacks import Callback
from data_files import *
import random

from sklearn.metrics.pairwise import cosine_similarity

import  re
#train_path = r"E:\zh\project\HPC\data\send\test\test1.csv"
train_paths = []
test_paths = []

train_datas = ['PARSEC', 'hpcc', 'hpl-s', 'hpcg', 'SPEC', 'GRAPH500', 'HPL-AI', 'LmBench', 'MiBench','RoyBench']
test_datas = ['SMG2000']

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

model_path = "./models/"#用于保存模型文件
if os.path.exists(model_path)==False:#帮助构造model保存路径
    os.mkdir(model_path, mode=0o777)
time_step = 3
data_scale = 10000

total_features = ['cpu-cycles', 'L1-dcache-load-misses', 'cache-misses', 'instructions', 'uncore_imc_0/cas_count_write/']
cpu_features = ['cpu-cycles', 'L1-dcache-load-misses', 'dTLB-loads', 'L1-dcache-loads', 'bus-cycles', 'ref-cycles', 'instructions', 'LLC-prefetches', 'L1-dcache-prefetch-misses', 'branch-loads']
mem_features = ['uncore_imc_0/cas_count_read/', 'uncore_imc_1/cas_count_read/', 'uncore_imc_2/cas_count_read/', 'uncore_imc_3/cas_count_read/', 'LLC-prefetch-misses',
                'node-prefetches', 'node-loads', 'LLC-load-misses', 'LLC-loads', 'cache-misses']
dense_features = cpu_features
#dense_features = ["instructions", "stalled-cycles-frontend", "uncore_imc_0/cas_count_write/", "uncore_imc_0/cas_count_read/", "cache-misses"]#该脚本中应使用十个PMC
targets = ["power/energy-pkg/"] #target用于定义被预测的目标是什么，可以选择切换为"power/energy-pkg/" 或者 下一行代码中的 "power/energy-ram/"
#targets = ["power/energy-ram/"]
DATA_DIM = 100  # 定义单个PMC事件embedding为100维度
interval_length = 100#每个PMC事件取值区间个数
lr = 0.001

#每个epoch结束计算在target上的error_rate
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

#构造GRU的输入
def create_dataset(dataset, label, look_back=time_step):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    dataNum = len(dataset)
    need_data = data_scale
    while need_data > 0:
        if need_data > dataNum - look_back - 2:
            dataX = dataX + dataset[:dataNum - look_back - 2].values.tolist()
            dataY = dataY + label[:dataNum - look_back - 2].tolist()
            need_data -= (dataNum - look_back - 2)
        else:
            dataX = dataX + dataset[:need_data].values.tolist()
            dataY = dataY + label[:need_data].tolist()
            need_data = 0
    return dataX, dataY

def error_rate_loss(true,pred):
    a = Lambda(lambda x: K.abs(x[0]-x[1]))([true,pred])
    error_rate = Lambda(lambda x: x[0]/x[1])([a,true])
    return error_rate

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
'''
def get_gru_model(data_shape):
    def error_rate_loss(true, pred):
        a = Lambda(lambda x: K.abs(x[0] - x[1]))([true, pred])
        error_rate = Lambda(lambda x: x[0] / x[1])([a, true])
        return error_rate

    def error_rate2_loss(true, pred):
        a = Lambda(lambda x: K.square(K.abs(x[0] - x[1])))([true, pred])
        error_rate = Lambda(lambda x: x[0] / x[1])([a, true])
        return error_rate

    def merge(input):
        tem = [input[i] for i in range(time_step)]
        # tem = [input[0],input[1],input[2]]
        return K.stack(tem, 1)  # 仅限于tems_step = 3时使用

    input1 = Input(shape=(data_shape + len(targets),), name="time_t1_input")
    input2 = Input(shape=(data_shape + len(targets),), name="time_t2_input")
    input3 = Input(shape=(data_shape + len(targets),), name="time_t3_input")
    edge_layer = Input(shape=(len(dense_features + targets), len(dense_features + targets)), name="edge_layer")

    gcn_model = get_gcn_model(data_shape)
    gcn_input1_cpu  = gcn_model([input1, edge_layer])
    gcn_input2_cpu  = gcn_model([input2, edge_layer])
    gcn_input3_cpu  = gcn_model([input3, edge_layer])
    tem_cpu = Lambda(merge, name="merge_cpu", output_shape=(time_step, DATA_DIM,))(
        [gcn_input1_cpu, gcn_input2_cpu, gcn_input3_cpu])


    ans_cpu = GRU(units=128, return_sequences=False, input_shape=(time_step, DATA_DIM,), name="gru_for_cpu")(tem_cpu)
    output_cpu = Dense(1)(ans_cpu)

    model = Model([input1, input2, input3, edge_layer], output_cpu, name="dynamic_model")

    # model.compile(optimizer=Adam(lr), loss="mse" )
    model.compile(optimizer=Adam(lr), loss=error_rate_loss)#loss 为error_rate_loss

    print(model.summary())
    return model
'''

def get_data(paths):
    result_data = []
    result_label = []
    print(paths)
    for path in paths:
        print(path)
        data = pd.read_csv(open(path))
        data["target"] = pd.Series(np.zeros(data["power/energy-pkg/"].shape[0]))
        special_node = ["target"]
        data, label = create_dataset(data[special_node + dense_features], np.array(data[targets]), time_step)
        print(len(data))
        result_data = result_data + data
        result_label = result_label + label
    print(len(result_data))
    return np.array(result_data), np.array(result_label)

def train():
    train_data, train_label = get_data(train_paths)#加载训练集
    test_data, test_label = get_data(test_paths)#加载测试集

    model = get_gru_model()

    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=2020)#训练集划分出一部分验证集
    #edge_matrix = get_edge_matrix()#获取图初始化矩阵，为对称阵
    #edge_matrixs = np.expand_dims(edge_matrix, axis=0)

    error_rate = Error_rate(
        X_val,#验证集数据
        y_val,#验证集ground-truth
        test_data,#测试集数据
        test_label,#测试集ground-truth
        targets[0],
    )

    model.fit(X_train,
              y_train, batch_size=512, epochs=15000, verbose=0, callbacks=[error_rate])
    # pred_ans = model.predict(test[dense_features+sparse_features], batch_size=256)


if __name__ == "__main__":
    train()
