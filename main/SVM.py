#coding=utf8
from sklearn import svm
# from LSTM import get_ranked_diff,read_Pc,select_fea
from yyh import get_ranked_diff_3,read_Pc,select_fea
from feature import *
import numpy as np
from keras.utils.np_utils import to_categorical

def get_X_Y_SVM(Pc,seq_len=10,test_percent=0.2):
    ranked_diff_series = get_ranked_diff_3(Pc)
    # print ranked_diff_series
    # ranked_diff = [v for v in ranked_diff_series]
    # print ranked_diff
    fea_yestPc = YEST(Pc)
    fea_yestPc = np.reshape(fea_yestPc, (fea_yestPc.shape[0], 1))
    Pc = np.array(Pc)
    fea_binary_ranked_diff = to_categorical(ranked_diff_series, 3)
    fea_ranked_diff = np.reshape(ranked_diff_series, (len(ranked_diff_series), 1))
    Ma6 = MA(4,Pc)
    fea_BIAS6 = BIAS(4,Pc,Ma6)
    fea_BIAS6 = np.reshape(fea_BIAS6, (fea_BIAS6.shape[0], 1))
    fea_MA5 = MA(3,Pc)
    fea_MA5 = np.reshape(fea_MA5, (fea_MA5.shape[0], 1))
    Sy = SY(Pc)
    fea_ASY4 = ASY(3,Sy)
    fea_ASY4 = np.reshape(fea_ASY4, (fea_ASY4.shape[0], 1))
    #                        0,1,2,3,4              5         6        7         8             9(y)
    fea_total = np.hstack((fea_binary_ranked_diff,fea_BIAS6,fea_MA5,fea_ASY4,fea_yestPc,fea_ranked_diff))
    X_total = []
    Y_total = []
    for i in range(len(fea_total)-1):
        X_total.append(fea_total[i])
        Y_total.append(fea_total[i+1,-1])
    X_total = np.array(X_total)
    Y_total = np.array(Y_total)
    train_len = int(len(X_total)*(1-test_percent))
    x_train = X_total[0:train_len]
    y_train = Y_total[0:train_len]
    x_test = X_total[train_len::]
    y_test = Y_total[train_len::]

    return x_train, y_train, x_test, y_test

def select_fea_SVM(x_train,x_test,indexs,seq_len=1):
    selected_X_train = x_train[:,indexs]
    selected_X_test = x_test[:,indexs]
    if seq_len == 1:
        return selected_X_train,selected_X_test
    head_X_train = [selected_X_train[0] for i in range(seq_len-1)]
    head_X_test = [selected_X_test[0] for i in range(seq_len-1)]
    selected_X_train = np.vstack((head_X_train,selected_X_train))
    selected_X_test = np.vstack((head_X_test,selected_X_test))
    seq_X_train = []
    seq_X_test = []
    for i in range(len(x_train)):
        seq_X_train.append(np.hstack(selected_X_train[i:i+seq_len]))
    for i in range(len(x_test)):
        seq_X_test.append(np.hstack(selected_X_test[i:i+seq_len]))

    return seq_X_train,seq_X_test

def cal_accuracy(predict,y):
    sum = 0.0
    for (v1,v2) in zip(predict,y):
        if v1 == v2:
            sum += 1
    return sum/len(y)

if __name__=='__main__':
    #[恒生指数，国企指数，上证综指，’沪深300‘，日经225，南韩综合，台湾加权，道琼斯，标普500，纳斯达克，德国dax，法国cac]
    index_code = ['^hsi','^hsce','000001.ss','000300.ss','^n225','^ks11','^twii','^dji','^gspc','^ixic','^gdaxi','^fchi']
    need_train = True
    Pc = read_Pc('^gspc')
    X_train, Y_train, X_test, Y_test = get_X_Y_SVM(Pc,seq_len=5,test_percent=0.2 )
    #
    # print 'X_train',X_train.shape
    # print 'Y_train', Y_train.shape
    # print 'X_test', X_test.shape
    # print 'Y_test', Y_test.shape

    # for i in range(100):
    new_X_train, new_X_test = select_fea_SVM(X_train,X_test,[0,1,2],seq_len = 5)
    clf = svm.SVC()  # class
    clf.fit(new_X_train, Y_train)  # training the svc model
    print new_X_train
    print "Y_train",Y_train
    result = clf.predict(new_X_test)
    accuray = cal_accuracy(result,Y_test)
    print 'SVM  seq_len=',5,'  ',accuray

