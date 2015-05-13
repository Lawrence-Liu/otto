__author__ = 'Zhao'
from src.classes.otclass import OttoProject
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from numpy.random import choice
op = OttoProject("../data/")
op.load_original()
X = op.subtrain_features
y = op.subtrain_classes
X = pd.DataFrame(np.log(X+1.0))
means = np.mean(X)
stds = np.std(X)
X = (X-means)/stds
n = X.shape[0]
vd = op.evaluate_features
vd = pd.DataFrame(np.log(vd+1))
vd = (vd-means)/stds

train = op.train_features
labels = op.train_features
test = op.test
train = np.log(train+1)
test = np.log(test+1)
center = np.mean(train)
scale = np.std(train)
train = (train-center)/scale
test = (test-center)/scale
n2 = train.shape[0]
for i in range(30, 130):
    index = choice(xrange(n),n,replace=True)
    new_X = X.copy().values[index]
    new_y = y.copy()[index]
    model = SVC(C=100.0, gamma=0.001, probability=True)
    model.fit(new_X, new_y)
    eva = model.predict_proba(vd)
    op.result = eva
    name = "eva_svm_"+str(i)+".csv"
    op.write_result(name=name)

    index2 = choice(xrange(n2),n2,replace=True)
    new_train = train.copy().values[index2]
    new_labels = labels.copy()[index2]
    model2 = SVC(C=100.0, gamma=0.001, probability=True)
    model2.fit(new_train, new_labels)
    pred = model.predict_proba(test)
    name2 = "bagging_svm_"+str(i)+".csv"
    op.result = pred
    op.write_result(name=name)
    print i

