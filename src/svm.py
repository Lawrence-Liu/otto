__author__ = 'Zhao'
"""
This svm script is designed to run on a single machine to test one set of tuning parameter each time.
2-fold cross validation will be used
"""
from src.classes.otclass import OttoProject
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random

def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss

#part 1
op = OttoProject('../data/', split_size=0.5)
op.load_original()
train = op.subtrain_features
train = pd.DataFrame(np.log(train+1))
means = np.mean(train)
stds = np.std(train)
train = (train-means)/stds  # centralization
train_label = op.subtrain_classes
test = op.evaluate_features
test_label = op.evaluate_classes
test = pd.DataFrame(np.log(test+1))
test = (test-means)/stds
model = SVC(probability=True, C=np.exp2(range(-5, 15))[0], gamma=np.exp2(range(-16, 9))[0])
model.fit(train, train_label)
pred_train = model.predict_proba(train)
pred_test = model.predict_proba(test)
l00 = logloss(train_label,pred_train)
l01 = logloss(test_label,pred_test)

#part 2

train = op.evaluate_features
train_label = op.evaluate_classes
test = op.subtrain_features
test_label =  op.subtrain_classes
train = pd.DataFrame(np.log(train+1))
means = np.mean(train)
stds = np.std(train)
train = (train-means)/stds  # centralization
test = pd.DataFrame(np.log(test+1))
test = (test-means)/stds
model.fit(train, train_label)
pred_train = model.predict_proba(train)
pred_test = model.predict_proba(test)
l10 = logloss(train_label, pred_train)
l11 = logloss(test_label, pred_test)

#write result

f = open(op.path+"log.txt","w")
f.write("model: "+str(model)+"\n")
f.write("train_logloss,test_logloss\n")
f.write(str(l00)+","+str(l01)+"\n")
f.write(str(l10)+","+str(l11)+"\n")
f.close()