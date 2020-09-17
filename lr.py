import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import linear_model 

dense_features = ["usr", "used_1", "used_2","csw", "int"]

target = ['result']
train_path = "../data/a_linpack.csv"
test_path = "../data/a_hpcg.csv"
root = "../models/lr/"

def get_data(path,dense_features,target):
    data = pd.read_csv(path)
    try:
        data["result"]=data["CPU_POWER"]
        pass
    except:
        data["result"]=data["CPU"]
    
    data[dense_features] = data[dense_features].fillna(0,)

    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    return np.array(data[dense_features]),np.array(data[target])
    
if __name__ == "__main__":

    data,label = get_data(train_path,dense_features,target)
    test_data,test_label = get_data(test_path,dense_features,target)

    if os.path.exists(root+'/')==False:
        os.mkdir(root+'/', mode=0o777)
    model = linear_model.LinearRegression()#加载线性回归模型
    model.fit(data, label)
    pred_ans = model.predict(test_data)

    error = 0
    for x,y in zip(test_label,pred_ans):
        x,y = x[0],y[0]
        error+=abs(float(x)-float(y))/float(x)
    test_error_rate = error/len(pred_ans)#计算error_rate
 
    test_mae = round(mean_absolute_error(test_label, pred_ans), 4)

    print("test_error_rate :",test_error_rate)
    print("test_mae :",test_mae)
    
    f = open(root+'/overall_metrics.txt',"w",encoding = "utf8")
    f.write("train_data: "+train_path+"\n")
    f.write("test_data: "+test_path+"\n")
    f.write("test_mae: "+str(test_mae)+"\n")
    f.write("test_error_rate: "+str(test_error_rate)+"\n")
    f.close()