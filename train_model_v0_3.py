
# coding: utf-8

# #采用单个时间点来做预测
# #上交的代码将读入data/15_sample_data.csv文件,由于其只包含15000个点,所以训练结果仅用作示范,没有实际意义

from __future__ import unicode_literals
import os
import warnings
from numpy import newaxis
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import h5py


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def conv2flag(number):
    if number == 133:
        return 1
    elif number == 132:
        return 1
    else:
        return 0

def conv2Nonelabel(number):
    if number == 1:
        return 1
    elif number == 2:
        return 0
    else:
        return np.nan

def conv2Noneflag(number):
    if number == 0:
        return np.nan
    else:
        return 1

def convert_dataset(x_data, y_data, look_back=60, sweep = 1):
    dataX, dataY = [], []
    for i in range(0,len(x_data)-look_back-1,sweep):
        a = x_data[i:(i+look_back)]
        dataX.append(a)
        dataY.append(y_data[i + look_back - 1])
    return np.array(dataX), np.array(dataY)

def drop_nan(dataX, dataY):
    dataX_new = []
    dataY_new = []
    for i in range(dataX.shape[0]):
        if str(dataY[i]) == 'nan':
            continue
        else:
            nan_flag = 0
            for j in range(dataX.shape[1]):
                for k in range(dataX.shape[2]):
                    if str(dataX[i,j,k]) == 'nan':
                        nan_flag = 1
                        break
                if nan_flag == 1:
                    break
            if nan_flag == 0:
                dataX_new.append(dataX[i])
                dataY_new.append(dataY[i])
    return np.array(dataX_new), np.array(dataY_new)

def get_dataFrame(num):
    #从本地的data文件夹读取训练数据
    df1=pd.read_csv('data/'+num+'_data.csv',sep=',') 
#     df2=pd.read_csv('data/'+num+'data-weather.csv',sep=',',header=None) 
#     df2.columns = ['temperature','relative_humidity','pressure']
#     df =pd.concat([df1,df2], axis = 1) 
    df = df1
    
    df['wtur_flt_main'] = df['wtur_flt_main'].diff()
    df['wtur_flt_main'] = df['wtur_flt_main'].map(conv2flag)
    df['wman_state'] = df['wman_state'].map(conv2Noneflag)
    qr = df.query('wtur_flt_main == 1')
    x_old = 1
    for x in qr.index :
        if (x - x_old) > 6000:
            #1 for fault surely;2 for health surely; 0 (default) for unsure
            df['wtur_flt_main'][(x-1000):x] = 1
            df['wtur_flt_main'][(x-6000):(x-4000)] = 2 #-12h to -8h is sure for health
        else:
            df['wtur_flt_main'][(x-1000):x] = 1 #1000 is for 2 hours (1.94h);        
        x_old = x
    df['wtur_flt_main'] = df['wtur_flt_main'].map(conv2Nonelabel)    
    return df

#获取温度差
def add_delta_tmp(df, delta):
    # delta = 6000: 12h
    col = 'tmp_%s'%(delta,)
    if col not in df.columns:
        df.insert(df.shape[1], col ,1)
    df[col] = df["environment_tmp"].diff(delta)
    return df

# get_data 与 get_data_sub为研究特征选取时所用的函数.
def get_data(df):
    # v0:包含全部原始的特征值，没有包含天气数据，测试准确率约为80%
    look_back = 60
    sweep= 2
    df_sel = df.drop(["time", "wtid", 'temperature','relative_humidity',
                        'pressure','wman_state','wtur_flt_main'], 1) 
    
    # v1:仅仅包含由人工选取出来的特征值，包含天气数据，取一个时间段,测试准确率约为70%
#     look_back = 60
#     sweep= 2
#     df_sel = df.loc[:,["wind_speed",
#             "generator_speed","power","wind_direction_mean",
#                 "yaw_position","pitch1_angle","pitch2_angle",
#                        "pitch3_angle",                
#                 "environment_tmp","int_tmp",
#                 'temperature','relative_humidity','pressure']]
    
    #输入新数据进行预测时,应该首先对输入数据按照原来的标准进行归一化
    column_min = {}
    column_max = {}
    for i in range(df_sel.shape[1]):
        column_min[list(df_sel)[i]] = np.min(df_sel.iloc[:,i])
        column_max[list(df_sel)[i]] = np.max(df_sel.iloc[:,i])
        
    norm_df = df_sel.iloc[:,:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    norm_df['wman_state'] = df['wman_state']
    
    x_data = norm_df.iloc[:,0:norm_df.shape[1]].values
    y_data = df.loc[:,'wtur_flt_main'].values # 'wtur_flt_main' 不可加方括号

    dataX, dataY = convert_dataset(x_data, y_data, look_back, sweep)
    dataX_new, dataY_new = drop_nan(dataX, dataY)
    dataX_new = dataX_new[:,:,0:-1]
    dataY_new = np_utils.to_categorical(dataY_new)
    
    end = dataX_new.shape[0]
    # end = 2000
    spl = 0.7
    X_train = dataX_new[0:int(end*spl)]
    y_train = dataY_new[0:int(end*spl)]
    X_test = dataX_new[int(end*spl):int(end*1)]
    y_test = dataY_new[int(end*spl):int(end*1)]
    print("=== get_data_v0 ===")
    print("dimensions of traing data and test data are:")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return [column_min, column_max, X_train, y_train, X_test, y_test]
    
#另一类获取训练数据的方式
# 先对df进行dropnan处理，再进行时间段切割，理论上来说这样做不够严谨，因为会造成
# 原本在时间上没有连在一起的数据连在一起, 但在look-back = 1时这个问题不存在.
def get_data_sub(df):
#     v0包含全部原始的特征值，没有包含天气数据
    look_back = 1
    sweep= 1
    df = add_delta_tmp(df, 3000)
#     df = add_delta_tmp(df, 3000)
#     df = add_delta_tmp(df, 1500)
    
    df_sel = df.dropna()
    y_data = df_sel.loc[:,'wtur_flt_main'].values # 'wtur_flt_main' 不可加方括号
#     df_sel = df_sel.drop(["time", "wtid", 'temperature','relative_humidity'
#                           ,'pressure','wman_state','wtur_flt_main'], 1)

#     df_sel = df_sel.drop(["time", "wtid", 'temperature','wman_state','wtur_flt_main'], 1)
    
#     df_sel = df_sel.loc[:,["wind_speed",
#             "generator_speed","power","wind_direction",
#             "wind_direction_mean",
#             "yaw_position","yaw_speed",
#             "pitch1_angle","pitch2_angle","pitch3_angle", 
#             "pitch1_speed","pitch2_speed","pitch3_speed",
#             "acc_x","acc_y",
#             "environment_tmp","int_tmp",
#             'relative_humidity']]
    
#     df_sel = df_sel.loc[:,["wind_speed",
#         "generator_speed","power","wind_direction_mean",
#             "yaw_position","pitch1_angle","pitch2_angle",
#                    "pitch3_angle",                
#             "environment_tmp","int_tmp",'relative_humidity','pressure']]
#     df_sel = df.loc[:,["wind_speed",
#     "power","wind_direction_mean",
#         "yaw_position","pitch1_angle","pitch2_angle",
#                "pitch3_angle",                
#         "environment_tmp",'relative_humidity','pressure']]

#     df_sel = df_sel.loc[:,["wind_speed",
#             "generator_speed","power","wind_direction",
#             "wind_direction_mean",
#             "yaw_position",
#             "pitch1_angle","pitch2_angle","pitch3_angle", 
#             "acc_x","acc_y",
#             "environment_tmp","int_tmp"
#             ]]

#最终的特征选取方案
    df_sel = df_sel.loc[:,["wind_speed",
        "generator_speed","power","wind_direction",
        "wind_direction_mean",
        "yaw_position",
        "pitch1_angle","pitch2_angle","pitch3_angle", 
        "environment_tmp","int_tmp",
        "acc_x","acc_y",
        "pitch1_ng5_tmp","pitch2_ng5_tmp","pitch3_ng5_tmp",
        "pitch1_moto_tmp","pitch2_moto_tmp","pitch3_moto_tmp",
        'tmp_3000'
        ]]

    #输入新数据进行预测时,应该首先对输入数据按照原来的标准进行归一化
    column_min = {}
    column_max = {}
    for i in range(df_sel.shape[1]):
        column_min[list(df_sel)[i]] = np.min(df_sel.iloc[:,i])
        column_max[list(df_sel)[i]] = np.max(df_sel.iloc[:,i])
        
    norm_df = df_sel.iloc[:,:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    x_data = norm_df.iloc[:,:].values
    
    dataX, dataY = convert_dataset(x_data, y_data, look_back, sweep)
    dataY = np_utils.to_categorical(dataY)
    
    end = dataX.shape[0]
    # end = 2000
    spl = 0.7
    X_train = dataX[0:int(end*spl)]
    y_train = dataY[0:int(end*spl)]
    X_test = dataX[int(end*spl):int(end*1)]
    y_test = dataY[int(end*spl):int(end*1)]
    print("=== get_data_sub resultes===")
    print("dimensions of traing data and test data are:")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return [column_min, column_max, X_train, y_train, X_test, y_test]

#计算混淆举证的四个指标
def metrics4(y_true1, y_pred1):
    y_true = np.zeros(y_true1.shape, dtype = int)
    y_pred = np.zeros(y_pred1.shape, dtype = int)
    for i in range(y_pred.shape[0]):
        y_pred[i] = y_pred1[i]
        y_true[i] = y_true1[i]
    N = len(y_true)
    TP = sum(y_true[:,1] & y_pred[:,1])
    TN = sum(y_true[:,0] & y_pred[:,0])
    FP = sum(y_true[:,0] & y_pred[:,1])
    FN = sum(y_true[:,1] & y_pred[:,0])
#     print(TP,TN,FP,FN)
    acc = (TP + TN)/N
    pre = TP/(TP + FP)
    rec = TP/(TP + FN)
    F1 = 2*TP/(2*TP + FP + FN)
    return [acc, pre, rec, F1]

def build_lstm_model(layers,look_back = 60):
    model = Sequential()
    
    model.add(LSTM(
        input_shape=(look_back,layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("softmax"))

    opt = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_cnn_model(layers, look_back = 60):
    model = Sequential()   
    
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=8,
        nb_col=layers[0],
        border_mode='same',     # Padding method
        input_shape=(1,look_back,layers[0])))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    # Padding method
    ))
    
    model.add(Convolution2D(64, 8, layers[0], border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("softmax"))

    opt = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    print("> Compilation Time : ", time.time() - start)
    return model

def get_edge(model, data):
    predicted = model.predict(data)
    f=[]
    for i in range(predicted.shape[0]):
        if predicted[i][0]<predicted[i][1]:
            f.append(i)
    return f
    #consider to get a edge if the state changes hold for 7 points
    #这样得到的应该是一个方波模样的东西;所谓的聚类中心,指的就是在那个点开始报故障
    
def plot_results(predicted_data, true_data,ep):
    #if ep%4 == 1:
    plt.clf()
    fig = plt.figure(facecolor='white')
    #fig2 = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    #bx = fig2. add_subplot(111)
    #ax.plot(true_data,color='red', label='True Data')
    ax.plot(predicted_data,'b*--', label = 'Predicted Break Point')
    ax.plot(true_data,'r',label='True Label')
    box=ax.get_position()
    ax.set_position(box)
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15))
    plt.xlabel('Index')
    plt.ylabel('Segmentation Attribute')
    #plt.savefig(NUM+'_'+str(ep)+'.png')
    #plt.show()
    
def plot(predicted_data):
    plt.clf()
    fig = plt.figure(facecolor='white')    
    ax = fig.add_subplot(111)
    ax.plot(predicted_data,'b')
    box=ax.get_position()
    ax.set_position(box)
    plt.show()

# to train an autoencoder
# best traning results for autoencoder: loss ~ 0.54
from keras.layers import Input, Dense
from keras.models import Model
def bulid_autoencoder(layers):
    input_img = Input(shape=(layers[0],))
    encoded = Dense(layers[1], activation='relu')(input_img)
    encoded = Dense(layers[2], activation='relu')(encoded)
    
    decoded = Dense(layers[1], activation='relu')(encoded)
    decoded = Dense(layers[0], activation='relu')(decoded)

    return Model(input_img, encoded), Model(input_img, decoded)

#训练 autoencoder的代码
# if __name__=='__main__':
#     print('> Loading data... ')
#     df = get_dataFrame('15')
#     df_sel = df.drop(["time","wtid",'wman_state','wtur_flt_main'], 1)
#     df_sel = df_sel.dropna()
#     norm_df = df_sel.iloc[:,:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

#     x_data = norm_df.iloc[:,:].values
#     encoder, autoencoder = bulid_autoencoder([x_data[1],1000,15])
#     print(encoder.summary())
# #     autoencoder.summary()
#     autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#     end = x_data.shape[0]
#     spl = 0.7
#     x_train = x_data[0:int(end*spl)]
#     x_test = x_data[int(end*spl):int(end*1)]

#     autoencoder.fit(x_train, x_train, epochs = 10, batch_size=256, shuffle=True,
#                    validation_data=(x_test, x_test))
#     cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
#     print('test cost: ', cost, 'test accuracy: ', accuracy)

#     x_data_encoded = encoder.predict(x_data)

# predict modle training
if __name__=='__main__':
    global_start_time = time.time()    
    epochs  = 1
    BATCH_SIZE = 10
    BATCH_INDEX = 0

    print('> Loading data... ')
    df = get_dataFrame('15_sample')
    
    #====LSTM
#     column_min, column_max, X_train, y_train, X_test, y_test = get_data(df)
    column_min, column_max, X_train, y_train, X_test, y_test = get_data_sub(df)
    
    print('> Data Loaded. Compiling...')
    model = build_lstm_model([X_train.shape[2], 40, 100, y_train.shape[1]], X_train.shape[1] )

#     # =====CNN
#     column_min, column_max, X_train, y_train, X_test, y_test = get_data(df)
#     X_train = np.expand_dims(X_train,axis=1)
#     X_test = np.expand_dims(X_test,axis=1)
    
#     f = open('tmp/column_min_cnn.txt','w')  
#     f.write(str(column_min))  
#     f.close() 
#     f = open('tmp/column_max_cnn.txt','w')  
#     f.write(str(column_max))  
#     f.close() 

#     print('> Data Loaded. Compiling...')
#     model = build_cnn_model([X_train.shape[3], 40, 100, y_train.shape[1]], X_train.shape[2] )

for ep in range(1,epochs+1):
    now = time.strftime("%y%m%d_%H%M")
    #保存训练所得模型
    filepath='tmp/lstm_weights_%s.hdf5' %(now,)
    #filepath='tmp/cnn_weights_%s.hdf5' %(now,)
    checkpointer = ModelCheckpoint(filepath, 
                                   verbose=1, save_best_only=True)
    #model.load_weights('tmp/lstm_weights_171223_2003.hdf5')   
    
    f = open('tmp/column_min.txt_%s'%(now,),'w')   
    f.write(str(column_min))  
    f.close() 
    f = open('tmp/column_max.txt_%s' %(now,),'w')  
    f.write(str(column_max))  
    f.close() 
    
    print('epoch=',ep)
    model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    nb_epoch=1,
    validation_split=0.02,
    callbacks=[checkpointer])
    
    cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
    print('test cost: ', cost, 'test accuracy: ', accuracy)
    y_pred = model.predict(X_test)
    
    y_pred_int = np.zeros(y_pred.shape, dtype = int)
    for i in range(y_pred.shape[0]):
        if y_pred[i][0] < y_pred[i][1]:
            y_pred_int[i][1] = 1
        else:
            y_pred_int[i][0] = 1
    print("[acc, pre, rec, F1] is:")
    print(metrics4(y_test,y_pred_int))
   
    y_pred_int = np.zeros(y_pred.shape[0], dtype = int)
    fault_pred = np.zeros(y_pred.shape[0], dtype = int)
    for i in range(y_pred.shape[0]):
        if y_pred[i][1] > 0.55:
            y_pred_int[i] = 1
    
    # 防抖措施，40 for about 5mins
    span = 40 * 3
    seuil = 0.9
    for k in range(span-1,y_pred.shape[0]-1,1):         
        if (sum(y_pred_int[k-span+1:k+1]) > seuil*span): 
            fault_pred[k] = 1

    print('Training duration (s) : ', time.time() - global_start_time)
    
#         plot_results(edg, y_test[:,1],ep)
    plt.clf()
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(fault_pred,'b*--', label = 'Predicted Fault Point')
    ax.plot(y_test[:,1],'r',label='True Label')
    box=ax.get_position()
    ax.set_position(box)
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15))
    plt.xlabel('Index')
    plt.ylabel('Segmentation Attribute')
    plt.show()

from influxdb import DataFrameClient
from influxdb import InfluxDBClient
# from train_model import *
from keras.models import load_model

def get_origin_data(num):
    df=pd.read_csv('data/'+num+'_data.csv',sep=',') 
    df['wman_state'] = df['wman_state'].map(conv2Noneflag)
       
    look_back = 1
    sweep= 1
    df = add_delta_tmp(df, 3000)
    
    df_sel = df.dropna()
    y_data = df_sel.loc[:,'wtur_flt_main'].values # 'wtur_flt_main' 不可加方括号
    time = df_sel.loc[:,'time'].values
    
    df_sel = df_sel.loc[:,["wind_speed",
        "generator_speed","power","wind_direction",
        "wind_direction_mean",
        "yaw_position",
        "pitch1_angle","pitch2_angle","pitch3_angle", 
        "environment_tmp","int_tmp",
        "acc_x","acc_y",
        "pitch1_ng5_tmp","pitch2_ng5_tmp","pitch3_ng5_tmp",
        "pitch1_moto_tmp","pitch2_moto_tmp","pitch3_moto_tmp",
        'tmp_3000'
        ]]
        
    norm_df = df_sel.iloc[:,:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    x_data = norm_df.iloc[:,:].values
    
    dataX, dataY = convert_dataset(x_data, y_data, look_back, sweep)
    dataY = np_utils.to_categorical(dataY)
    
    print("=== get_origin_data ===")
    print("dimensions data are:")
    print(dataX.shape)
    print(dataY.shape)
    print(time.shape)
    return [time, dataX, dataY]

#读入本地数据,将模型的输出批量写入到数据库中
# if __name__=='__main__':
#     db = 'windturbine'
#     client = InfluxDBClient(url_of_server, 8086, username, passwd, db )
#
#     [time, dataX, dataY] = get_origin_data('15_sample')
#     y_pred = model.predict(dataX)
#     y_pred_int = np.zeros(y_pred.shape[0], dtype = int)
#     fault_pred = np.zeros(y_pred.shape[0], dtype = int)
#     for i in range(y_pred.shape[0]):
#         if y_pred[i][1] > 0.55:
#             y_pred_int[i] = 1
#
#     # 40 for about 5mins
#     span = 40 * 3
#     seuil = 0.9
#     for k in range(span-1,y_pred.shape[0]-1,1):
#         if (sum(y_pred_int[k-span+1:k+1]) > seuil*span):
#             fault_pred[k] = 1
#
#     json_body = []
#     for i in range(dataX.shape[0]):
#         json_body.append(
#         {
#             "measurement": "diagnose_test1",
#             "tags": {
#                 "turbine": '150015'
#             },
#             "time": time[i],
#             "fields": {
#                 "fault_prob": y_pred[i,1],
#                 "fault_point": y_pred_int[i],
#                 "if_alert": fault_pred[i],
#             }
#         }
#             )
#         if i % 10000 == 0:
#             client.write_points(json_body)
#             json_body = []
