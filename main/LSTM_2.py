#coding=utf-8
import pandas as pd
import time
import csv
import pylab as plt
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,GRU
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from feature import *


def create_model(layers):
    """
    Returns a sequential model by given topological arguments.
    :param layers:[input_shape, hidden_dim, output_dim]
    :return:
    """
    model = Sequential()

    model.add(LSTM(
        input_shape=layers[0],  # need change
        output_dim=layers[1],
        return_sequences=False))
    model.add(Dropout(0.2))
    #
    # model.add(LSTM(
    #     output_dim=layers[2],
    #     return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(
    #     output_dim=layers[2],
    #     return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(
    #     output_dim=layers[2],
    #     return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("softmax"))

    start = time.time()
    print "Compilation Time : ", time.time() - start
    return model


def get_ranked_diff_5(Pc):
    diff1 = Pc.diff(1)
    diff1[0] = 0
    sorted_diff1 = sorted(diff1)
    Q1 = sorted_diff1[len(sorted_diff1)/5]
    Q2 = sorted_diff1[len(sorted_diff1)/5*2]
    Q3 = sorted_diff1[len(sorted_diff1)/5*3]
    Q4 = sorted_diff1[len(sorted_diff1)/5*4]
    print Q1,Q2,Q3,Q4
    ranked_diff1 = []
    for val in diff1:
        if val<Q1:
            ranked_diff1.append(0)
        elif val>=Q1 and val<Q2:
            ranked_diff1.append(1)
        elif val>=Q2 and val<Q3:
            ranked_diff1.append(2)
        elif val>=Q3 and val<Q4:
            ranked_diff1.append(3)
        else:
            ranked_diff1.append(4)
    return ranked_diff1

def get_ranked_diff_2(Pc):
    diff1 = Pc.diff(1)
    diff1[0] = 0
    sorted_diff1 = sorted(diff1)
    Q1 = sorted_diff1[len(sorted_diff1)/2]
    ranked_diff1 = []
    for val in diff1:
        if val<Q1:
            ranked_diff1.append(0)
        else:
            ranked_diff1.append(1)
    return ranked_diff1

def select_fea(x_train,x_test,indexs):
    return x_train[:,:,indexs],x_test[:,:,indexs]


def read_Pc(name):
    csvfile = file(name+'.csv', 'rb')
    reader = csv.reader(csvfile)
    Pc = []
    for line in reader:
        Pc.append(float(line[0]))
    Pc = pd.Series(Pc)
    return Pc


def get_X_Y(Pc,seq_len=10,test_percent=0.2):
    ranked_diff_series_5 = get_ranked_diff_5(Pc)
    ranked_diff_series_2 = get_ranked_diff_2(Pc)
    print 'hahahahahahahaa'
    #print ranked_diff_series_5
    # print ranked_diff_series
    # ranked_diff = [v for v in ranked_diff_series]
    # print ranked_diff
    fea_yestPc = YEST(Pc)
    fea_yestPc = np.reshape(fea_yestPc, (fea_yestPc.shape[0], 1))
    Pc = np.array(Pc)
    fea_binary_ranked_diff_5 = to_categorical(ranked_diff_series_5, 5)#5 dim
    fea_binary_ranked_diff_2 = to_categorical(ranked_diff_series_2, 2)  # 5 dim
    Ma6 = MA(6,Pc)
    print Ma6
    fea_BIAS6 = BIAS(6,Pc,Ma6)
    fea_BIAS6 = np.reshape(fea_BIAS6, (fea_BIAS6.shape[0], 1))
    fea_MA5 = MA(5,Pc)
    fea_MA5 = np.reshape(fea_MA5, (fea_MA5.shape[0], 1))
    Sy = SY(Pc)
    fea_ASY4 = ASY(5,Sy)#(4311,1)
    fea_ASY4 = np.reshape(fea_ASY4, (fea_ASY4.shape[0], 1))
    #print fea_binary_ranked_diff_5.shape#(4311,5)
    #                        0,1,2,3,4              5         6        7         8              9,10
    fea_total = np.hstack((fea_binary_ranked_diff_5,fea_BIAS6,fea_MA5,fea_ASY4,fea_yestPc,fea_binary_ranked_diff_2))
    #print fea_total.shape    (4311,11)
    seq_data = []
    for i in range(len(fea_total)-seq_len+1):
        seq_data.append(fea_total[i:i+seq_len])
    total_len = len(seq_data)
    seq_data = np.array(seq_data)
    train_len = int(total_len*(1-test_percent))
    x_train = seq_data[0:train_len,0:-1,:]
    y_train = seq_data[0:train_len,-1,0:5]
    x_test = seq_data[train_len::,0:-1,:]
    y_test = seq_data[train_len::,-1,0:5]

    return x_train, y_train, x_test, y_test


#[恒生指数，国企指数，上证综指，’沪深300‘，日经225，南韩综合，台湾加权，道琼斯，标普500，纳斯达克，德国dax，法国cac]
if __name__=='__main__':
    index_code = ['^hsi','^hsce','000001.ss','000300.ss','^n225','^ks11','^twii','^dji','^gspc','^ixic','^gdaxi','^fchi']
    need_train = True
    Pc = read_Pc('^n225')
    X_train, Y_train, X_test, Y_test = get_X_Y(Pc,seq_len=5,test_percent=0.2 )
    X_train, X_test = select_fea(X_train,X_test,[0,1,2,3,4])
    scores = []
    for i in range(10):
        model = create_model([(4, 5),100, 5])#input_shape=[seq_len-1,indicator_num]
        if need_train == False:
            model.load_weights("my_model.h5")
        else:
            op = RMSprop(lr=0.0005)
            model.compile(loss="categorical_crossentropy", optimizer=op, metrics=['accuracy'])
            model.fit(X_train, Y_train, nb_epoch=50, batch_size=150, verbose=1)
            model.save_weights('my_model.h5')

        score = model.evaluate(X_test, Y_test, verbose=0)
        scores.append((i + 1, score[1]))
        print 'score', score#loss,acc
    for s in scores:
        print s[0], s[1]
# scores = []
# for i in range(10):
#     model.fit(X_train, Y_train, nb_epoch=(i+1)*10, batch_size=10)
#     score = model.evaluate(X_test, Y_test)
#     scores.append(score)
# np.save('scores.npy',scores)

# plt.plot(range(100),ranked_diff1[0:100],'r-')
# plt.ylim(-3,3)
# plt.show()


