import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from random import sample
#from sklearn.decomposition import PCA

def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss

train_data = pd.read_csv('../data/train.csv')
rindex = np.array(sample(train_data.index, train_data.shape[0] * 2 / 3))
traindata = train_data.iloc[rindex,:]
testdata = train_data.drop(rindex)
train_label = traindata.target
traindata = traindata.drop(['target', 'id'], axis=1)
test_label = testdata.target
testdata = testdata.drop(['target', 'id'], axis=1)

gb_otto = GradientBoostingClassifier(n_estimators = 300, verbose = 2)
gb_otto.fit(traindata, train_label)
test_prob = gb_otto.predict_proba(testdata)
train_prob = gb_otto.predict_proba(traindata)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)

