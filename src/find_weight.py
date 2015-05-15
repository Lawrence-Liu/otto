__author__ = 'Zhao'
"""
This script is designed to find the optimal weight for the combination of ensembling different algorithms.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from classes.otclass import OttoProject
from sklearn.preprocessing import LabelBinarizer
op = OttoProject("../data/")
op.load_original()

def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss
eva_dl_1 = []
eva_svm = []
eva_xgb_05_05_01_8 = []
eva_xgb_01_444 = []
eva_xgb_sub_shr = []
eva_nn_default = []
eva_xgb_sub = []
eva_xgb_05_01_10 = []
eva_xgb_e_01_10 = []
eva_xgb_05_05_01_10 = []
eva_xgb_05_05_01_12 = []
fin = open('../eva_weight/eva_dl1.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    del line[0]
    del line[0]
    line = map(float, line)
    eva_dl_1.append(line)
fin.close()

fin = open('../eva_weight/eva_svm_100-0.001.csv','r')
fin.readline()
for line in fin.readlines():
    line = line.strip().split(',')
    del line[0]
    line = map(float, line)
    eva_svm.append(line)
fin.close()

fin = open('../eva_weight/eva_xgb_0.5_0.5_0.1_8.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_05_05_01_8.append(line)
fin.close()

fin = open('../eva_weight/eva_xgb_0.1_444.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_01_444.append(line)
fin.close()

fin = open('../eva_weight/eva_xgb_sub_0.1.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_sub_shr.append(line)
fin.close()

fin = open('../eva_weight/eva_nn_19.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_nn_default.append(line)
fin.close()
fin = open('../eva_weight/eva_xgb_sub0.5.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_sub.append(line)
fin.close()
fin = open('../eva_weight/eva_xgb_max10_0.1_0.5_180.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_05_01_10.append(line)
fin.close()
fin = open('../eva_weight/eva_xgb_e_0.1_10.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_e_01_10.append(line)
fin.close()
fin = open('../eva_weight/eva_xgb_0.5_0.5_10.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_05_05_01_10.append(line)
fin.close()
fin = open('../eva_weight/eva_xgb_0.5_0.5_12.csv','r')
for line in fin.readlines():
    line = line.strip().split(',')
    line = map(float, line)
    eva_xgb_05_05_01_12.append(line)
fin.close()

eva_xgb_sub_shr=np.array(eva_xgb_sub_shr[1:12377])
eva_xgb_05_05_01_8 = np.array(eva_xgb_05_05_01_8[1:12377])
eva_xgb_01_444 = np.array(eva_xgb_01_444[1:12377])
eva_dl_1 = np.array(eva_dl_1)
eva_svm = np.array(eva_svm[1:12377])
eva_nn_default = np.array(eva_nn_default[1:12377])
eva_xgb_sub = np.array(eva_xgb_sub[1:12377])
eva_xgb_05_01_10 = np.array(eva_xgb_05_01_10[1:12377])
eva_xgb_e_01_10 = np.array(eva_xgb_e_01_10[1:12377])
eva_xgb_05_05_01_10 = np.array(eva_xgb_05_05_01_10[1:12377])
eva_xgb_05_05_01_12 = np.array(eva_xgb_05_05_01_12[1:12377])

vd_label = op.evaluate_classes[1:12377]

a,b,c,d,e,f,g,h,i,j = 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0
k = 0.0
sig = 1

fout = open("weight_change.txt","w")
while sig == 1:
    base = logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                             eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*k)/100.0)
    if logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                        eva_xgb_sub*g+eva_xgb_05_01_10*(h-0.5)+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base \
            and h>0:
        h -= 0.5
        k += 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*(j-0.5)+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base and j>0:
        j -= 0.5
        k += 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*(i-0.5)+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base and i>0:
        i -= 0.5
        k += 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*(f-0.5)+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base and f>0:
        f -= 0.5
        k += 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*(a-0.5)+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                        eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base and a>0:
        a -= 0.5
        k += 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*(e-0.5)+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) < base and e>0:
        e -= 0.5
        k += 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*(b-0.5)+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) and b>0:
        b -= 0.5
        k += 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*(d-0.5)+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) and d>0:
        d -= 0.5
        k += 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*(c-0.5)+eva_xgb_01_444*d+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) and c>0:
        c -= 0.5
        k += 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                          eva_xgb_sub*(g-0.5)+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k+0.5))/100.0) and g > 0:
        g -= 0.5
        k += 0.5


    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                        eva_xgb_sub*g+eva_xgb_05_01_10*(h+0.5)+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base \
            and k>0:
        h += 0.5
        k -= 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*(j+0.5)+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base and k>0:
        j += 0.5
        k -= 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*(i+0.5)+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base and k>0:
        i += 0.5
        k -= 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*(f+0.5)+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base and k>0:
        f += 0.5
        k -= 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*(a+0.5)+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                        eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base and k>0:
        a += 0.5
        k -= 0.5
    elif logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*(e+0.5)+eva_nn_default*f+
                                   eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) < base and k>0:
        e += 0.5
        k -= 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*(b+0.5)+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) and k>0:
        b += 0.5
        k -= 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*(d+0.5)+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) and k>0:
        d += 0.5
        k -= 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*(c+0.5)+eva_xgb_01_444*d+eva_xgb_sub_shr*e+
                                          eva_nn_default*f+eva_xgb_sub*g+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) and k>0:
        c += 0.5
        k -= 0.5
    elif base > logloss(vd_label,(eva_xgb_05_05_01_8*a+eva_svm*b+eva_dl_1*c+eva_xgb_01_444*d+eva_xgb_sub_shr*e+eva_nn_default*f+
                                          eva_xgb_sub*(g+0.5)+eva_xgb_05_01_10*h+eva_xgb_e_01_10*i+eva_xgb_05_05_01_10*j+eva_xgb_05_05_01_12*(k-0.5))/100.0) and k > 0:
        g += 0.5
        k -= 0.5

    else:
        sig = 0
    fout.write(str([a,b,c,d,e,f,g,h,i,j,k])+"\n")
print [a,b,c,d,e,f,g,h,i,j,k]
print base
fout.close()