import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from random import sample

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


rf_otto = RandomForestClassifier(n_estimators = 100, n_jobs = -1, max_features=30)
rf_otto.fit(traindata.iloc[:,1:94], traindata.iloc[:, 94])
test_prob = rf_otto.predict_proba(testdata.iloc[:, 1:94])
train_prob = rf_otto.predict_proba(traindata.iloc[:, 1:94])
print 'The logloss score of test data:', logloss(testdata.iloc[:, 94], test_prob)
print 'The logloss score of train data:', logloss(traindata.iloc[:, 94], train_prob)
