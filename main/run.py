import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pylab as pl
def bulid_X_Y(data,seq_len,test_len):
    X = []
    Y = []
    x_test = []
    y_test = []
    data_len = len(data)
    for i in range(data_len-seq_len-test_len):
        X.append(data[i:i+seq_len,:])
        Y.append(data[i+seq_len,3])
    for i in range(test_len):
        x_test.append(data[data_len-test_len-seq_len+i:data_len-test_len+i,:])
        y_test.append(data[data_len-test_len+i,3])
    return np.array(X), np.array(Y), np.array(x_test), np.array(y_test)

def normalised_data(data):
    nor_data = []
    cols = data.shape[1]
    for col_index in range(cols):
        new_col = [p/(data[0,col_index]) for p in data[:,col_index]]
        if nor_data == []:
            nor_data = new_col
        else:
            nor_data = np.vstack((nor_data,new_col))
    return np.array(nor_data).T

def load_data(filename,seq_len,test_len,normallised):
    import csv
    csvfile = file(filename, 'rb')
    reader = csv.reader(csvfile)
    data = []
    closing_price_original = []
    for i,line in enumerate(reader):
        data.append([float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4])])
        closing_price_original.append(float(line[3]))
    csvfile.close() 
    data = np.array(data)
    if normallised == True:
        data = normalised_data(data)
    X, Y, x, y = bulid_X_Y(data, seq_len, test_len)
    return X, Y, x, y,closing_price_original
    
    
def create_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print "Compilation Time : ", time.time() - start
    return model

def average_error(y_predict, y_true):
    sum_error = 0
    for yp,yt in zip(y_predict,y_true):
        sum_error =  sum_error + abs(yp-yt)/yt
    return sum_error/len(y_predict)*100
    
if __name__=='__main__':
    need_train = True
    X, Y, x_test, y_test, closing_price_original = load_data('ssec_big.csv', 6, 10, True)
#     X_original, Y_original, x_test_original, y_test_original = load_data('ssec.csv', 10, 10, False)
    model = create_model([5,10,20,1])
    if need_train == False: 
        model.load_weights("my_model.h5")
    else:
        model.fit(X, Y, nb_epoch=100, batch_size=10)
        model.save_weights('my_model.h5')

    y_predict = model.predict(x_test)
    days = range(10)
    first_day_cp = closing_price_original[0]
    y_predict_unnormalised = [p*first_day_cp for p in y_predict]
    y_test_unnormalised = [p*first_day_cp for p in y_test]
    ave_error = average_error(y_predict,y_test_unnormalised)
    print y_predict_unnormalised
    print y_test_unnormalised
    print ave_error
    pl.plot(days, y_test, 'r')
    pl.plot(days, y_predict, 'g')
    pl.show()
#     score = model.evaluate(x_test, y_test, batch_size=10)
