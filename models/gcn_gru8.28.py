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

train_datas = ['PARSEC', 'hpcc', 'hpl-s', 'hpcg', 'SPEC', 'GRAPH500', 'HPL-AI', 'LmBench','MiBench','RoyBench','SMG2000']
test_datas = []
test_No = 2
train_datas = train_datas[:test_No]+train_datas[test_No+1:]
test_datas = train_datas[test_No:test_No+1]

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
#mem_features = ['uncore_imc_0/cas_count_read/','uncore_imc_0/cas_count_write/','LLC-prefetch-misses','node-prefetches', 'node-loads', 'LLC-load-misses', 'LLC-loads', 'cache-misses',
#                'instructions','L1-dcache-load-misses']
dense_features = mem_features
#dense_features = ["instructions", "stalled-cycles-frontend", "uncore_imc_0/cas_count_write/", "uncore_imc_0/cas_count_read/", "cache-misses"]#该脚本中应使用十个PMC
#targets = ["power/energy-pkg/"] #target用于定义被预测的目标是什么，可以选择切换为"power/energy-pkg/" 或者 下一行代码中的 "power/energy-ram/"
targets = ["power/energy-ram/"]
DATA_DIM = 100  # 定义单个PMC事件embedding为100维度
interval_length = 100#每个PMC事件取值区间个数
lr = 0.001


#把PMC数值映射到其区间编号
def mapPMCs(data):
    intervals = open("..\\data\\pmc_intervals.txt", encoding="utf8").readlines()
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
    return data

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
        #print(epoch)
        results = "\t"
        error = 0
        result = self.model.predict(self.val_data)
        for x, y in zip(self.val_label, result):
            error += abs(x - float(y)) / float(x)
        error_rate = error / len(self.val_label)
        results = results + str(mean_absolute_error(self.val_label, result)) + "\t" + str(error_rate) + "\t"
        logs[self.target +'val mean_absolute_error'] = mean_absolute_error(self.val_label, result)
        #print(self.target+ "val mean_absolute_error: %f" % (mean_absolute_error(self.val_label, result)))
        logs[self.target+' val_error_rate'] = error_rate
        #print(self.target +" val_error_rate: %f" % (error_rate))
        improve = True
        if error_rate >= self.error_rate:#只有当验证集上的Eerror_rate降低以后，才对测试集进行测试，进而保存模型文件
            improve = False
        self.error_rate = error_rate
        self.model.save(model_path+"/error_rate_"+str(error_rate)+"_.h5")
        error = 0
        result = self.model.predict(self.test_data)
        for x, y in zip(self.test_label, result):
            error += abs(x - float(y)) / float(x)
        test_error_rate = error / len(self.test_label)
        logs[self.target +'test mean_absolute_error'] = mean_absolute_error(self.test_label, result)
        results = results + '\t' + str(mean_absolute_error(self.test_label, result))
        #print(self.target+ "test mean_absolute_error: %f" % (mean_absolute_error(self.test_label, result)))
        logs[self.target+' test_error_rate'] = test_error_rate
        results = results + '\t' + str(test_error_rate)
        if not improve:
            results = results + '\t' + 'No improve'
        #print(self.target +" test_error_rate: %f" % (test_error_rate))
        print(epoch, results)

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


def get_gcn_model(data_shape):  # 静态。edge_layer是描述边的矩阵（N*N），data_layer是描述点的矩阵(N*feature_dim)
    def extract(input):
        return input[:, 0, :]  #返回input[:,0,:]为 target 功耗

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
    input6 = Lambda(slice, name="slice_feature6", output_shape=(1,), arguments={'index': 6})(input_layer)
    input7 = Lambda(slice, name="slice_feature7", output_shape=(1,), arguments={'index': 7})(input_layer)
    input8 = Lambda(slice, name="slice_feature8", output_shape=(1,), arguments={'index': 8})(input_layer)
    input9 = Lambda(slice, name="slice_feature9", output_shape=(1,), arguments={'index': 9})(input_layer)
    input10 = Lambda(slice, name="slice_feature10", output_shape=(1,), arguments={'index': 10})(input_layer)
    # input_dim 取决于每个PMC的区间长度
    embed0 = Embedding(input_dim=1, output_dim=DATA_DIM, input_length=1, trainable=True)(input0)
    embed1 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input1)
    embed2 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input2)
    embed3 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input3)
    embed4 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input4)
    embed5 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input5)
    embed6 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input6)
    embed7 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input7)
    embed8 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input8)
    embed9 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input9)
    embed10 = Embedding(input_dim=interval_length, output_dim=DATA_DIM, input_length=1, trainable=True)(input10)
    embed = Lambda(merge, name="merge", output_shape=(data_shape + len(targets), DATA_DIM,))(
        [embed0, embed1, embed2, embed3, embed4, embed5, embed6, embed7, embed8, embed9, embed10])

    edge_layer = Input(shape=(len(dense_features + targets), len(dense_features + targets)))
    conv_layer = GraphConv(
        units=DATA_DIM,
        step_num=1,
    )([embed, edge_layer])
    # tem = Dense(32)(conv_layer)
    tem_cpu = Lambda(extract, name="extract_cpu", output_shape=(DATA_DIM,))(conv_layer)
    #tem_cpu = Extract(5)(conv_layer)
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

    print(model.summary())
    return model

def get_edge_matrix():
    tem = pd.read_csv(r".\InitialWeight.csv")  #
    col = list(tem.columns)
    try:
        col.remove('Unnamed: 0')
    except:
        pass
    print([col.index(each) for each in targets+dense_features])
    edge_matrix = np.array(tem[targets + dense_features])[[col.index(each) for each in targets + dense_features]]
    print(edge_matrix)
    edge_matrix = np.round(edge_matrix)  # 四舍五入保留到个位
    # print(edge_matrix)
    return edge_matrix#为对称矩阵
def get_data(paths):
    result_data = []
    result_label = []
    print(paths)
    for path in paths:
        print(path)
        data = pd.read_csv(open(path))
        data["target"] = pd.Series(np.zeros(data["power/energy-pkg/"].shape[0]))
        data = mapPMCs(data)
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

    model = get_gru_model(len(dense_features))

    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=2020)#训练集划分出一部分验证集
    edge_matrix = get_edge_matrix()#获取图初始化矩阵，为对称阵
    edge_matrixs = np.expand_dims(edge_matrix, axis=0)

    error_rate = Error_rate(
        [X_val[:, 0, :], X_val[:, 1, :], X_val[:, 2, :], edge_matrixs.repeat(X_val.shape[0], axis=0)],#验证集数据
        y_val,#验证集ground-truth
        [test_data[:, 0, :], test_data[:, 1, :], test_data[:, 2, :], edge_matrixs.repeat(test_data.shape[0], axis=0)],#测试集数据
        test_label,#测试集ground-truth
        targets[0],
    )

    model.fit([X_train[:, 0, :], X_train[:, 1, :], X_train[:, 2, :], edge_matrixs.repeat(X_train.shape[0], axis=0)],
              y_train, batch_size=512, epochs=15000, verbose=0, callbacks=[error_rate])
    # pred_ans = model.predict(test[dense_features+sparse_features], batch_size=256)


if __name__ == "__main__":
    train()
