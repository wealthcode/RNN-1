#coding=utf-8
import math
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
# import pandas.io.data as web
import pandas_datareader.data as web
import math
import csv
'''
#从雅虎财经获取DAX指数的数据
index_code = ['^gdaxi']
# index_code = ['^hsi','^hsce','000001.ss','000300.ss','^n225','^ks11','^twii','^dji','^gspc','^ixic','^gdaxi','^fchi']
for name in index_code:
    print '1'
    Index = web.DataReader(name=name, data_source='yahoo',start = '1990-1-1',end = '2017-1-1')
    print '2'
    print Index



    Pc1 = list(Index['Open'])   #0
    Pc1 = np.array(Pc1)
    Pc2 = list(Index['Close'])#1
    Pc2 = np.array(Pc2)
    Pc3 = list(Index['Volume'])#2
    Pc3 = np.array(Pc3)
    Pc4 = list(Index['High'])#3
    Pc4 = np.array(Pc4)
    Pc5 = list(Index['Low'])#4
    Pc5 = np.array(Pc5)
    c=np.stack((Pc1, Pc2,Pc3,Pc4,Pc5), axis=-1)
    out = open(name+'_k.csv', 'w')
    csv_writer = csv.writer(out)
    csv_writer.writerows(c)
'''
csvfile = file('^gdaxi_k.csv', 'r')
reader = csv.reader(csvfile)
sp500 = []
out = open('^gdaxi_knn.csv', 'w')
csv_writer = csv.writer(out)
for line in reader:
    if(line[0]>=line[1]):
	line.append(0)#5
	line.append(float(line[1])-float(line[3]))#6
    if(line[1]>line[0]):
	line.append(1)
	line.append(float(line[1])-float(line[4]))
    csv_writer.writerow(line)
print 'finished'

#sp500 = pd.Series(sp500)

