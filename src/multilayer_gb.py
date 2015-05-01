import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
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

data = pd.read_csv('/Users/lawrence/Dropbox/kaggle/otto/data/train.csv')
data_23 = data[(data.target == 'Class_2') | (data.target == 'Class_3')]
data = data.replace({'Class_2' : 'Class_2_3', 'Class_3' : 'Class_2_3'})
rindex = np.array(sample(data.index, data.shape[0] * 2 / 3))

train_data = data.loc[rindex,:]
test_data = data.drop(rindex)
train_label = train_data.target
train_data = train_data.drop(['target', 'id'], axis=1)
test_label = test_data.target
test_data = test_data.drop(['target', 'id'], axis=1)

gb_otto = GradientBoostingClassifier(n_estimators = 100, verbose = 1, warm_start = 1)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)
train_prob = gb_otto.predict_proba(train_data)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)

gb_otto.set_params(n_estimators = 150)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)
train_prob = gb_otto.predict_proba(train_data)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)

gb_otto.set_params(n_estimators = 200)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)
train_prob = gb_otto.predict_proba(train_data)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)


gb_otto.set_params(n_estimators = 250)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)
train_prob = gb_otto.predict_proba(train_data)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)

gb_otto.set_params(n_estimators = 300)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)
train_prob = gb_otto.predict_proba(train_data)
print 'The logloss score of test data:', logloss(test_label, test_prob)
print 'The logloss score of train data:', logloss(train_label, train_prob)