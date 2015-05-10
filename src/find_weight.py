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
eva_rf = []
eva_svm = []
eva_xgb_444 = []
eva_xgb_sub = []
eva_xgb_sub_shr = []
eva_nn_default = []
f = open('../eva_weight/eva_rf_1000.csv','r')
f.readline()
for l in f.readlines():
    l = l.strip().split(',')
    del l[0]
    l = map(float, l)
    eva_rf.append(l)
f.close()

f = open('../eva_weight/eva_svm_100-0.001.csv','r')
f.readline()
for l in f.readlines():
    l = l.strip().split(',')
    del l[0]
    l = map(float, l)
    eva_svm.append(l)
f.close()

f = open('../eva_weight/eva_xgb_0.1_444.csv','r')
for l in f.readlines():
    l = l.strip().split(',')
    l = map(float, l)
    eva_xgb_444.append(l)
f.close()

f = open('../eva_weight/eva_xgb_sub0.5.csv','r')
for l in f.readlines():
    l = l.strip().split(',')
    l = map(float, l)
    eva_xgb_sub.append(l)
f.close()

f=open('../eva_weight/eva_sub_0.1.csv','r')
for l in f.readlines():
    l = l.strip().split(',')
    l = map(float, l)
    eva_xgb_sub_shr.append(l)
f.close()

f = open('../eva_weight/eva_nn_19.csv','r')
for l in f.readlines():
    l = l.strip().split(',')
    l = map(float, l)
    eva_nn_default.append(l)
f.close()

eva_xgb_sub_shr=np.array(eva_xgb_sub_shr)

eva_xgb_444 = np.array(eva_xgb_444)
eva_xgb_sub = np.array(eva_xgb_sub)
eva_rf = np.array(eva_rf)
eva_svm = np.array(eva_svm)
eva_nn_default = np.array(eva_nn_default)
vd_label = op.evaluate_classes
a = 40.0
b = 8.0
c = 2.0
d = 5.0
e = 45.0
f = 0.0
sig = 1
while sig == 1:
    base = logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*f)/100)
    if base > logloss(vd_label,(eva_xgb_444*(a-1.0)+eva_svm*b+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f+1.0))/100.0) and a>0:
        a = a-1.0
        f = f+1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*(e-1.0)+eva_nn_default*(f+1.0))/100.0) and e>0:
        e = e-1.0
        f = f+1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*(b-1.0)+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f+1.0))/100.0) and b>0:
        b = b-1.0
        f = f+1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*c+eva_xgb_sub*(d-1.0)+eva_xgb_sub_shr*e+eva_nn_default*(f+1.0))/100.0) and d>0:
        d = d-1.0
        f = f+1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*(c-1.0)+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f+1.0))/100.0) and c>0:
        c = c-1.0
        f = f+1.0

    elif base > logloss(vd_label,(eva_xgb_444*(a+1.0)+eva_svm*b+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f-1.0))/100.0) and f>0:
        a = a+1.0
        f = f-1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*(e+1.0)+eva_nn_default*(f-1.0))/100.0) and f>0:
        e = e+1.0
        f = f-1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*(b+1.0)+eva_rf*c+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f-1.0))/100.0) and f>0:
        b = b+1.0
        f = f-1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*c+eva_xgb_sub*(d+1.0)+eva_xgb_sub_shr*e+eva_nn_default*(f-1.0))/100.0) and f>0:
        d = d+1.0
        f = f-1.0
    elif base > logloss(vd_label,(eva_xgb_444*a+eva_svm*b+eva_rf*(c-1.0)+eva_xgb_sub*d+eva_xgb_sub_shr*e+eva_nn_default*(f+1.0))/100.0) and c>0:
        c = c+1.0
        f = f-1.0

    else:
        sig = 0
print [a,b,c,d,e,f]
print base