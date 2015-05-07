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



def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss


op = OttoProject("../data/")
op.load_original()
train = op.subtrain_features
vd = op.evaluate_features
train = pd.DataFrame(np.log(train+1))
vd = pd.DataFrame(np.log(vd+1))
means = np.mean(train)
stds = np.std(train)
train = (train-means)/stds
vd = (vd-means)/stds

model1 = SVC(C=100, gamma=0.001, probability=True)
model2 = GBC(n_estimators=300, verbose=2)
model3 = RFC(n_estimators=1000, n_jobs=3)
model1.fit(train, op.subtrain_classes)
model2.fit(train, op.subtrain_classes)
model3.fit(train, op.subtrain_classes)
eva1 = model1.predict_proba(vd)
eva2 = model2.predict_proba(vd)
eva3 = model3.predict_proba(vd)

ll = [[],[]]
for x in range(0,101):
    for y in range(0,101-x):
        ll[0].append(logloss(op.evaluate_classes,eva1*x/100.0+eva2*y/100.0+eva3*(1-x/100.0-y/100.0)))
        ll[1].append([x,y])
minimum = min(ll[0])
index = ll[0].index(minimum)
weight = [ll[1][index][0],ll[1][index][1],100-ll[1][index][0]-ll[1][index][1]]
print weight