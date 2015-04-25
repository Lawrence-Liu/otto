import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from random import sample

#input
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
train_label = train_data.target
train_data = train_data.drop(['id', 'target'], axis = 1)
test_data = test_data.drop('id', axis = 1)


#train model
gb_otto = GradientBoostingClassifier(n_estimators = 300, verbose = 2)
gb_otto.fit(train_data, train_label)
test_prob = gb_otto.predict_proba(test_data)


#generate output
output  = pd.DataFrame(test_prob)
index = np.arange(1, 144369)
index.shape = (144368, 1)
output.insert(loc = 0, column = 'index', value = index )
output.to_csv('../data/submission.csv', index = False, float_format = '%1.5f',
	header=['id', 'Class_1', ,'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
