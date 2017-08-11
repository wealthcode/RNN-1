from sklearn import svm
from sklearn import preprocessing
from feature import *
from run import *
import pylab as plt
X, Y, x_test, y_test, closing_price_original = load_data('ssec_big.csv', 3, 10, True)
closing_price_original = closing_price_original[-40::]
test_len = 10
train_len = len(closing_price_original) - test_len
days = range(test_len)
Pc = closing_price_original
Sy = SY(Pc)
Asy4 = ASY(4,Sy)
Ma5 = MA(5,Pc)
Ma6 = MA(6,Pc)
Bias6 = BIAS(6,Pc,Ma6)
X_test = []
for i in range(test_len):
    X_test.append([Bias6[i+train_len-1],Ma5[i+train_len-1],Asy4[i+train_len]])

# Pc = closing_price_original[0:30]
# Sy = SY(Pc)
# Asy4 = ASY(4,Sy)
# Ma5 = MA(5,Pc)
# Ma6 = MA(6,Pc)
# Bias6 = BIAS(6,Pc,Ma6)
X_train = []
Y_train = []
for i in range(train_len):
    if i!=0:
        X_train.append([Bias6[i-1],Ma5[i-1],Asy4[i]])
        Y_train.append(Pc[i])
svr_model=svm.SVR(C=1024,gamma=0.5)
X_train = preprocessing.scale(X_train)
svr_model.fit(X_train,Y_train)
X_test = preprocessing.scale(X_test)
y_predict = svr_model.predict(X_test)
# print 'X_train',X_train
# print "Y_train",Y_train
# print "X_test",X_test
print y_predict
print closing_price_original[train_len::]
ave_error = average_error(y_predict, closing_price_original[train_len::])
print ave_error

plt.plot(days,closing_price_original[train_len::],'r*-')
plt.plot(days,y_predict,'b*-')
# plt.ylim(1240, 1300)
plt.show()