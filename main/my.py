#coding=utf-8
import math
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
# import pandas.io.data as web
import pandas_datareader.data as web
import math
import csv

#从雅虎财经获取DAX指数的数据
index_code = ['^n225']
# index_code = ['^hsi','^hsce','000001.ss','000300.ss','^n225','^ks11','^twii','^dji','^gspc','^ixic','^gdaxi','^fchi']
for name in index_code:
    # Index=[]

    Index = web.DataReader(name=name, data_source='yahoo',start = '1990-1-1',end = '2017-1-1')
    Pc1 = list(Index['Open'])
    Pc1 = np.array(Pc1)
    Pc2 = list(Index['Close'])
    Pc2 = np.array(Pc2)
    # Pc3 = list(Index['High'])
    # Pc3 = np.array(Pc3)
    # Pc4 = list(Index['Low'])
    # Pc4 = np.array(Pc4)
    c=np.stack((Pc1, Pc2), axis=-1)
    out = open(name+'_use.csv', 'w')
    csv_writer = csv.writer(out)
    csv_writer.writerows(c)
    print name,' finished'
'''

name='^n225'
csvfile = file(name+'_use.csv', 'r')
reader = csv.reader(csvfile)
sp500 = []
out = open(name+'_new.csv', 'w')
csv_writer = csv.writer(out)
for line in reader:
    if(line[0]>=line[1]):
	line.append(0)
    if(line[1]>line[0]):
	line.append(1)
    csv_writer.writerow(line)
print 'finished'

#sp500 = pd.Series(sp500)
#print sp500'''
