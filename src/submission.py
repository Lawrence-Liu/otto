import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from random import sample



train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
train_label = train_data.target
train_data = train_data.drop(['id', 'target'], axis = 1)
test_data = test_data.drop('id', axis = 1)

#tfidf
tf_idf = TfidfTransformer()
train_data = tf_idf.fit_transform(train_data).toarray()
test_data = tf_idf.fit_transform(test_data).toarray()

#predict probability
rf_otto = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
rf_otto.fit(train_data, train_label)
test_prob = rf_otto.predict_proba(test_data)

#generate output
output  = pd.DataFrame(test_prob)
index = np.arange(1, 144369)
index.shape = (144368, 1)
output.insert(loc = 0, column = 'index', value = index )
output.to_csv('../data/submission.csv', index = False)
