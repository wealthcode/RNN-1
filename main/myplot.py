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

csvfile = file('new.csv', 'rb')
reader = csv.reader(csvfile)
y = []
i=0
x=[]
for line in reader:
    x.append(i)
    y.append(float(line[1]))
    i=i+1
plt.figure(figsize=(8,4)) #创建绘图对象
plt.plot(x,y,linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("Time") #X轴标签
plt.ylabel("Price")  #Y轴标签
plt.show()  #显示图
plt.savefig("line.jpg") #保存图
print 'h'