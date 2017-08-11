'''
Created on 2017-3-6

@author: Administrator
'''
import pylab as plt
from run import *
from feature import *
X, Y, x_test, y_test, closing_price_original = load_data('ssec.csv', 3, 10, True)
days = range(10)
Pc = closing_price_original[30::]
Sy = SY(Pc)
Asy5 = ASY(5,Sy)
Ma5 = MA(5,Pc)
Ma6 = MA(6,Pc)
Bias6 = BIAS(6,Pc,Ma6)
print Pc
print days
# plt.figure('1')
plt.plot(days,Pc,'r*-')
plt.plot(days,Ma5,'b*-')
plt.ylim(1240, 1300)
# plt.figure('2')
# plt.plot(days,Sy,'r')
# plt.figure('3')
# plt.plot(days,Bias6,'r')
plt.show()
