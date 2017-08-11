import  math
import numpy as np
# def MA(num,Pc):
#     head = [Pc[0] for i in range(num-1)]
#     extendPc = head + Pc
#     Ma = []
#     for i in range(len(Pc)):
#         Ma.append(sum(extendPc[i:i+num])/num)
#     return Ma

def MA(num,Pc):#incluing forecast day
    head = np.array([Pc[0] for i in range(num-1)])
    extendPc = np.hstack((head,Pc))
    Ma = []
    for i in range(len(Pc)):
        Ma.append(sum(extendPc[i:i+num])/num)
    return np.array(Ma)

def YEST(Pc):
    one = np.array(Pc[0])
    extendPc = np.hstack((one, Pc[0:-1]))
    return extendPc


def ASY(num,Sy):#incluing forecast day
    head = np.array([Sy[0] for i in range(num-1)])
    extendSy = np.hstack((head,Sy))
    Asy = []
    for i in range(len(Sy)):
        Asy.append(sum(extendSy[i:i+num])/num)
    return np.array(Asy)

def BIAS(num,Pc,Ma):#incluing forecast day
    Bias = []
    for pc,ma in zip(Pc,Ma):
        Bias.append(pc*((pc-ma)/ma)*100)
    return np.array(Bias)

def SY(Pc):#incluing forecast day
    Sy = []
    for i in range(len(Pc)):
        if i == 0:
            Sy.append(0)
        else:
            Sy.append((math.log(Pc[i],math.e)-math.log(Pc[i-1],math.e))*100)
    return np.array(Sy)

# Pc = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
# print 'Pc',Pc
# Sy = SY(Pc)
# print 'Sy',Sy
# Asy5 = ASY(5,Sy)
# print 'Asy5',Asy5
# Ma5 = MA(5,Pc)
# print 'Ma5',Ma5
# Ma6 = MA(6,Pc)
# print 'Ma6',Ma6
# Bias6 = BIAS(6,Pc,Ma6)
# print 'Bias6',Bias6



