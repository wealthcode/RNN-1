#coding=utf8
import csv
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt

csvfile = file('sp500.csv', 'rb')
reader = csv.reader(csvfile)
sp500 = []
for line in reader:
    sp500.append(float(line[0]))
sp500 = pd.Series(sp500)
from statsmodels.tsa.stattools import adfuller as ADF
adf_result = ADF(sp500)
print 'adf:',adf_result[0]
print 'p:',adf_result[1]
print 'c_value',adf_result[4]
# print u'原始序列的ADF检验结果为：', ADF(sp500)
diff1 = sp500.diff(1)
diff1[0] = 0
adf_result_diff1 = ADF(diff1)
print 'adf:',adf_result_diff1[0]
print 'p:',adf_result_diff1[1]
print 'c_value',adf_result_diff1[4]

plot_acf(sp500[len(sp500)-101::]).show()
plot_pacf(sp500[len(sp500)-101::]).show()

dta = sp500[len(sp500)-100::]
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2100'))
print dta
model = ARIMA(dta, (1,1,1)).fit()
predict_sunspots = model.predict('2090', '2100', dynamic=True)
print(predict_sunspots)
plt.show()