import pandas as pd
import keras
import keras.backend as K
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#print(gpus)

#tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
#    ])
#print(tf.config.experimental.get_virtual_device_configuration(gpus[0]))
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from sklearn.model_selection import train_test_split
from keras_gcn import GraphConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import Callback
#from keras.utils.vis_utils import plot_model

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import os, re


#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz 2.44.1\bin"

time_step = 3
#dense_features = ["instructions", "stalled-cycles-frontend", "uncore_imc_0/cas_count_write/",
#                  "uncore_imc_0/cas_count_read/", "cache-misses"]
dense_features = ['cpu-cycles', 'L1-dcache-load-misses', 'cache-misses', 'instructions', 'uncore_imc_0/cas_count_write/']
#targets = ["power/energy-cores/"] #target用于定义被预测的目标是什么
targets = ["power/energy-ram/"]
DATA_DIM = 100  # 定义单个PMC事件embedding为100维度
interval_length = 100#每个PMC事件取值区间个数
lr = 0.001


#把PMC数值映射到其区间编号
def mapPMCs(data):
    intervals = open("../data/pmc_intervals.txt", encoding="utf8").readlines()
    for line in intervals:
        for pmc in dense_features:
            if pmc in line:
                threholds = re.findall("\([0-9\.]+, [0-9\.]+\)", line)
                pairs = []
                for each in threholds:
                    tem = each.replace("(", "").replace(")", "").replace(" ", "").split(",")
                    tem = (float(tem[0]), float(tem[1]))
                    pairs.append(tem)
                a = list(data[pmc])
                # print(pairs)
                # print(a)
                converted_a = []
                for each in a:
                    for index, interval in enumerate(pairs):
                        if each >= interval[0] and each < interval[1]:
                            converted_a.append(index)
                #print(converted_a)
                data[pmc] = converted_a
            # print(data[dense_features])
    return data

#每个epoch结束计算在target上的error_rate
class Error_rate(Callback):
    def __init__(self, data, label,target):
        super(Error_rate, self).__init__()
        self.data = data
        self.label = label
        self.target = target
    def on_epoch_end(self, epoch, logs=None):
        error = 0
        # result = self.model.predict([self.data[:,0,:],self.data[:,1,:],self.data[:,2,:]])
        result = self.model.predict(self.data)
        for x, y in zip(self.label, result):
            error += abs(x - float(y)) / float(x)
        error_rate = error / len(self.label)
        logs[self.target +' mean_absolute_error'] = mean_absolute_error(self.label, result)
        print(self.target+ " mean_absolute_error: %f" % (mean_absolute_error(self.label, result)))
        logs[self.target+' train_error_rate'] = error_rate
        print(self.target +" train_error_rate: %f" % (error_rate))


#构造GRU的输入
def create_dataset(dataset, label, look_back=time_step):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a.values)
        dataY.append(label[i + look_back - 1])
    print(dataX[0])
    #print(type(dataY[0]))
    #print(dataY[0])
    return np.array(dataX), np.array(dataY)


def get_gcn_model(data_shape):  # 静态。edge_layer是描述边的矩阵（N*N），data_layer是描述点的矩阵(N*feature_dim)
    def extract(input):
        return input[:, 5, :]  #返回input[:,0,:]为 cpu功耗，返回input[:,1,:]为dram功耗

    def merge(input):
        tem = [input[i] for i in range(data_shape + len(targets))]
        return K.stack(tem, 1)

    def slice(input, index):#取出所有样本某一维度的所有值
        return input[:, index]

    input_layer = Input(shape=(data_shape + len(targets),))
    input0 = Lambda(slice, name="slice_cpu", output_shape=(1,), arguments={'index': 0})(input_layer)
    input1 = Lambda(slice, name="slice_feature1", output_shape=(1,), arguments={'index': 1})(input_layer)
    input2 = Lambda(slice, name="slice_feature2", output_shape=(1,), arguments={'index': 2})(input_layer)
    input3 = Lambda(slice, name="slice_feature3", output_shape=(1,), arguments={'index': 3})(input_layer)
    input4 = Lambda(slice, name="slice_feature4", output_shape=(1,), arguments={'index': 4})(input_layer)
    input5 = Lambda(slice, name="slice_feature5", output_shape=(1,), arguments={'index': 5})(input_layer)
    # input_dim 取决于每个PMC的区间长度
    embed0 = Embedding(input_dim=1, output_dim=DATA_DIM, input_length=1, trainable=True)(input0)
    embed1 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input1)
    embed2 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input2)
    embed3 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input3)
    embed4 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input4)
    embed5 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input5)
    embed = Lambda(merge, name="merge", output_shape=(data_shape + len(targets), DATA_DIM,))(
        [embed0, embed1, embed2, embed3, embed4, embed5])

    edge_layer = Input(shape=(len(dense_features + targets), len(dense_features + targets)))
    conv_layer = GraphConv(
        units=DATA_DIM,
        step_num=1,
    )([embed, edge_layer])
    # tem = Dense(32)(conv_layer)
    tem_cpu = Lambda(extract, name="extract_cpu", output_shape=(DATA_DIM,))(conv_layer)

    model = Model([input_layer, edge_layer], tem_cpu, name="static_model")
    #plot_model(model, to_file='../subs/static_model_structure.png', show_shapes=True)

    print(model.summary())
    return model


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
    #plot_model(model, to_file='../subs/dynamic_model_structure.png', show_shapes=True)

    print(model.summary())
    return model

  # 需要构造边矩阵，作为静态传入参数,这里随机初始化了边的关系，尚未使用PMC相关性矩阵
def get_edge_matrix():
    tem = pd.read_csv(r".\InitialWeight.csv")  #
    col = list(tem.columns)
    try:
        col.remove('Unnamed: 0')
    except:
        pass
    print(targets+dense_features)
    print([col.index(each) for each in targets+dense_features])
    edge_matrix = np.array(tem[targets + dense_features])[[col.index(each) for each in targets + dense_features]]
    print(edge_matrix)
    edge_matrix = np.round(edge_matrix)  # 四舍五入保留到个位
    # print(edge_matrix)
    return edge_matrix#为对称矩阵


def train():
    #get_edge_matrix()
    #exit()
    data = pd.read_csv(r"..\..\SPEC2017\1\2020-07-21-11-42-01-speed-int-test.csv")
    data["target"] = pd.Series(np.zeros(data["power/energy-cores/"].shape[0]))
    data = mapPMCs(data)
    #print(data[dense_features])
    special_node = ["target"]

    data, label = create_dataset(data[special_node + dense_features], np.array(data[targets]), time_step)
    #print(data[0])
    model = get_gru_model(len(dense_features))
    #plot_model(model, to_file='../subs/model_structure.png', show_shapes=True)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2020)
    edge_matrix = get_edge_matrix()
    edge_matrixs = np.expand_dims(edge_matrix, axis=0)

    error_rate = Error_rate(
        [X_test[:, 0, :], X_test[:, 1, :], X_test[:, 2, :], edge_matrixs.repeat(X_test.shape[0], axis=0)],
        y_test,targets[0])

    model.fit([X_train[:, 0, :], X_train[:, 1, :], X_train[:, 2, :], edge_matrixs.repeat(X_train.shape[0], axis=0)],
              y_train, batch_size=512, epochs=15000, verbose=1, callbacks=[error_rate])
    # pred_ans = model.predict(test[dense_features+sparse_features], batch_size=256)


if __name__ == "__main__":
    train()
