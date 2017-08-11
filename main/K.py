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
from keras.callbacks import EarlyStopping
from lossHistory import LossHistory



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
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        output_dim=layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

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
        output_dim=layers[3]))
    model.add(Activation("softmax"))
    tim=time.time()
    print 'start time:',time.ctime(tim)
    return model


def get_ranked_diff_3(Pc):
    ranked_diff1=[]
    print 'yyh'
    print Pc
    print Pc.__len__()
    change=[]
    for i in range(Pc.__len__()):
        # if i==1:
        #     ranked_diff1.append(2)
        # else:
        #     if Pc[i]-Pc[i-1]>1:
        #         ranked_diff1.append(0)
        #     elif Pc[i-1]-Pc[i]>1:
        #         ranked_diff1.append(1)
        #     else:
        #         ranked_diff1.append(2)
        if i==1:
            ranked_diff1.append(2)
            change.append(0)
        else:
            if Pc[i]>Pc[i-1]:
                ranked_diff1.append(0)
            elif Pc[i-1]>Pc[i]:
                ranked_diff1.append(1)
            else:
                ranked_diff1.append(2)
            change.append(Pc[i] - Pc[i - 1])
    return ranked_diff1,change
    '''
    diff1 = Pc.diff(1)
    diff1[0] = 0
    sorted_diff1 = sorted(diff1)
    # Q1 = sorted_diff1[len(sorted_diff1)/5]
    # Q2 = sorted_diff1[len(sorted_diff1)/5*2]
    # Q3 = sorted_diff1[len(sorted_diff1)/5*3]
    # Q4 = sorted_diff1[len(sorted_diff1)/5*4]
    # print Q1,Q2,Q3,Q4
    # ranked_diff1 = []
    # for val in diff1:
    #     if val<Q1:
    #         ranked_diff1.append(0)
    #     elif val>=Q1 and val<Q2:
    #         ranked_diff1.append(1)
    #     elif val>=Q2 and val<Q3:
    #         ranked_diff1.append(2)
    #     elif val>=Q3 and val<Q4:
    #         ranked_diff1.append(3)
    #     else:
    #         ranked_diff1.append(4)
    # return ranked_diff1
    Q1 = sorted_diff1[len(sorted_diff1) / 3]
    Q2 = sorted_diff1[len(sorted_diff1) / 3* 2]
    print Q1, Q2
    ranked_diff1 = []
    for val in diff1:
        if val < Q1:
            ranked_diff1.append(0)
        elif val >= Q1 and val < Q2:
            ranked_diff1.append(1)
        elif val >= Q2:
            ranked_diff1.append(2)
    return ranked_diff1
    '''
# def get_ranked_diff_2(Pc):
#     diff1 = Pc.diff(1)
#     diff1[0] = 0
#     sorted_diff1 = sorted(diff1)
#     Q1 = sorted_diff1[len(sorted_diff1)/2]
#     ranked_diff1 = []
#     for val in diff1:
#         if val<Q1:
#             ranked_diff1.append(0)
#         else:
#             ranked_diff1.append(1)
#     return ranked_diff1

def select_fea(x_train,x_test,indexs):
    return x_train[:,:,indexs],x_test[:,:,indexs]


def read_Pc(name):
    csvfile = file('huangjin.csv', 'rb')
    print 'hahahahahahahaa'
    print name
    reader = csv.reader(csvfile)
    Pc = []
    # for line in reader:
    #     Pc.append(float(line[1]))
    return Pc


def get_X_Y(Pc,seq_len=10,test_percent=0.2):
    ranked_diff_series_3,changed = get_ranked_diff_3(Pc)
    fey= []
    k=[]
    vol=[]
    out = open('huangjin.csv')
    csv_writer = csv.reader(out)
    # for line in csv_writer:
    #     #print len(fey)
    #     vol.append(line[2])
    #     fey.append(line[5])
    #     k.append(line[6])
        # print line[2]
    # print fey
    # ranked_diff_series_2 = get_ranked_diff_2(Pc)
    # print ranked_diff_series
    # ranked_diff = [v for v in ranked_diff_series]
    # print ranked_diff
    fea_yestPc = YEST(Pc)
    fea_yestPc = np.reshape(fea_yestPc, (fea_yestPc.shape[0], 1))
    Pc = np.array(Pc)
    fey_binary_ranked_diff_2 = to_categorical(fey, 2)
    fea_k=np.array(k)
    fea_k=np.reshape(fea_k,(fea_k.shape[0], 1))
    vol = np.array(vol)
    fea_vol = np.reshape(vol, (vol.shape[0], 1))
    changed = np.array(changed)
    fea_changed = np.reshape(changed, (changed.shape[0], 1))
    print 'xixixixi'
    print fey_binary_ranked_diff_2.shape
    fea_binary_ranked_diff_3 = to_categorical(ranked_diff_series_3, 3)
    print fea_binary_ranked_diff_3.shape#5 dim
    # fea_binary_ranked_diff_2 = to_categorical(ranked_diff_series_2, 2)  # 5 dim
    Ma6 = MA(6,Pc)
    fea_BIAS6 = BIAS(6,Pc,Ma6)
    fea_BIAS6 = np.reshape(fea_BIAS6, (fea_BIAS6.shape[0], 1))
    fea_MA5 = MA(5,Pc)
    fea_MA5 = np.reshape(fea_MA5, (fea_MA5.shape[0], 1))
    Sy = SY(Pc)
    fea_ASY4 = ASY(4,Sy)
    # print fea_ASY4.shape#(4311,1)
    fea_ASY4 = np.reshape(fea_ASY4, (fea_ASY4.shape[0], 1))
    #print fea_binary_ranked_diff_5.shape#(4311,5)
    #                        0,1,2              3         4       5        6      7                       8
    fea_total = np.hstack((fea_binary_ranked_diff_3,fea_changed,fea_BIAS6,fea_k,fea_vol,fey_binary_ranked_diff_2,fea_MA5,fea_ASY4,fea_yestPc))
    print 'yyhyyhyyh'
    print fea_total    #(4311,11)
    seq_data = []
    for i in range(len(fea_total)-seq_len+1):
        seq_data.append(fea_total[i:i+seq_len])
    total_len = len(seq_data)
    seq_data = np.array(seq_data)
    print seq_data.shape
    print seq_data
    train_len = int(total_len*(1-test_percent))
    x_train = seq_data[0:train_len,0:-1,:]
    y_train = seq_data[0:train_len,-1,0:3]
    x_test = seq_data[train_len::,0:-1,:]
    y_test = seq_data[train_len::,-1,0:3]
    # print 'yyhyyhyyhyhh'
    # print x_train
    # print 'stop'
    # print y_train 121 211
    return x_train, y_train, x_test, y_test


#[恒生指数，国企指数，上证综指，’沪深300‘，日经225，南韩综合，台湾加权，道琼斯，标普500，纳斯达克，德国dax，法国cac]
if __name__=='__main__':
    index_code = ['^hsi','^hsce','000001.ss','000300.ss','^n225','^ks11','^twii','^dji','^gspc','^ixic','^gdaxi','^fchi']
    need_train = True
    Pc = read_Pc('^dji')#Pc = pd.Series(Pc)
    X_train, Y_train, X_test, Y_test = get_X_Y(Pc,seq_len=5,test_percent=0.2 )
    X_train, X_test = select_fea(X_train,X_test,[0,1,2,3])
    scores = []
    losshistory=LossHistory()
    for i in range(5):
        model = create_model([(4,4),50,100, 3])#input_shape=[seq_len-1,indicator_num]
        if need_train == False:
            model.load_weights("my_model.h5")
        else:
            op = RMSprop(lr=0.0005)
            model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy'])
            # model.fit(X_train, Y_train, nb_epoch=50, batch_size=150, verbose=1)
            # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            model.fit(X_train, Y_train, validation_split=0.2, callbacks=[losshistory])
            model.save_weights('my_model.h5')

        score = model.evaluate(X_test, Y_test, verbose=0)
        scores.append((i + 1, score[1]))
        print 'score', score#loss,acc
    sum=0
    for s in scores:
        print s[0], s[1]
        sum=sum+s[1]
    losshistory.loss_plot('epoch')
    print "average=",sum/5
    print "7 feature 5day"
# scores = []
# for i in range(10):
#     model.fit(X_train, Y_train, nb_epoch=(i+1)*10, batch_size=10)
#     score = model.evaluate(X_test, Y_test)
#     scores.append(score)
# np.save('scores.npy',scores)

# plt.plot(range(100),ranked_diff1[0:100],'r-')
# plt.ylim(-3,3)
# plt.show()


