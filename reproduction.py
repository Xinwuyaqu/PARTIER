# 导入 keras 等相关包
import pandas as pd
import numpy as np
from data_files import *
from matplotlib import pyplot as plt
import random
from numpy import mat

from sklearn.metrics.pairwise import cosine_similarity

import  re
#train_path = r"E:\zh\project\HPC\data\send\test\test1.csv"
train_paths = []
test_paths = []

train_datas = ['PARSEC', 'hpcc', 'hpl-s', 'hpcg', 'SPEC', 'GRAPH500', 'HPL-AI', 'LmBench','MiBench','RoyBench','SMG2000']
test_datas = []
test_No = 0
train_datas = train_datas[:test_No]+train_datas[test_No+1:]
test_datas = train_datas[test_No:test_No+1]

train_datas = ['ref']
test_datas = ['test']

for data_file in data_files:
    if not 'SPEC' in data_file: continue
    if not 'speed' in data_file:continue
    if 'HPCC' in data_file:
        if not 'single' in data_file:
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

features = ['instructions','stalled-cycles-frontend','cpu-cycles','LLC-load-misses']

def create_dataset(dataset, label):
    # 这里的look_back与timestep相同
    nos = [random.randint(0, len(dataset) - 2) for p in range(0,10000)]
    return [dataset[i] for i in nos],[label[i] for i in nos]

def get_data(paths):
    result_data = []
    result_label = []
    print(paths)
    for path in paths:
        #print(path)
        data = pd.read_csv(open(path))
        data, label = create_dataset(data[features].values.tolist(), data[['power/energy-pkg/','power/energy-ram/']].values.tolist())
        #print(len(data))
        result_data = result_data + data
        result_label = result_label + label
    print(len(result_data))
    return result_data, result_label

train_data,train_label = get_data(train_paths)
test_data,test_label = get_data(test_paths)
cpu_train=[p[0] for p in train_label]
mem_train=[p[1] for p in train_label]
cpu_test=[p[0] for p in test_label]
mem_test=[p[1] for p in test_label]

train_uops = [p[0]/p[2] for p in train_data]
train_actv = [(p[2]-p[1])/p[2] for p in train_data]
train_l3ms = [p[3]/p[2] for p in train_data]
train_l3ms2= [p*p for p in train_l3ms]

test_uops = [p[0]/p[2] for p in test_data]
test_actv = [(p[2]-p[1])/p[2] for p in test_data]
test_l3ms = [p[3]/p[2] for p in test_data]
test_l3ms2= [p*p for p in test_l3ms]

# CPU
train_datas = mat([[1]*len(train_uops),train_actv,train_uops])
transpose=train_datas.transpose()
inv = np.linalg.inv(train_datas*transpose)
coeff=inv*train_datas*(mat(cpu_train).transpose())
test_datas = mat([[1]*len(test_uops),test_actv,test_uops])
cpu_predict=(coeff.transpose()*test_datas).tolist()[0]
cpu_real = cpu_test
average_error = sum([abs(p-q)/q for p,q in zip(cpu_predict, cpu_real)])/len(cpu_real)
print(average_error)
cpu_real = cpu_train
average_error = sum([abs(p-q)/q for p,q in zip(cpu_predict, cpu_real)])/len(cpu_real)
print(average_error)

# MEM
train_datas = mat([[1]*len(train_l3ms),train_l3ms,train_l3ms2])
transpose=train_datas.transpose()
inv = np.linalg.inv(train_datas*transpose)
coeff=inv*train_datas*(mat(mem_train).transpose())
test_datas = mat([[1]*len(test_l3ms),test_l3ms,test_l3ms2])
mem_predict=(coeff.transpose()*test_datas).tolist()[0]
mem_real = mem_test
average_error = sum([abs(p-q)/q for p,q in zip(mem_predict, mem_real)])/len(mem_real)
print(average_error)
mem_real = mem_train
average_error = sum([abs(p-q)/q for p,q in zip(mem_predict, mem_real)])/len(mem_real)
print(average_error)