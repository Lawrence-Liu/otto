from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from src.classes.otclass import OttoProject
'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.

    Compatible Python 2.7-3.4 

    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python keras_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.

    Best validation score at epoch 21: 0.4881 

    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

np.random.seed(111) # for reproducibility

def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss

def load_data_2(path, train=True):
    op = OttoProject(path)
    if train:
        train = pd.DataFrame(op.subtrain_features).astype(np.float32)
        labels = pd.DataFrame(op.subtrain_classes)
        return train, labels
    else:
        test = pd.DataFrame(op.test).astype(np.float32)
        return test


def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
op = OttoProject('../data/')
op.load_original()
X = pd.DataFrame(op.subtrain_features).astype(np.float32)
labels = pd.DataFrame(op.subtrain_classes)
vd = pd.DataFrame(op.evaluate_features).astype(np.float32)
vd_labels = pd.DataFrame(op.evaluate_classes)
X_test = pd.DataFrame(op.test).astype(np.float32)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, _ = preprocess_data(X_test, scaler)
vd, _ = preprocess_data(vd, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

model = Sequential()
model.add(Dense(dims, 512, init='glorot_uniform'))
model.add(PReLU((512,)))
model.add(BatchNormalization((512,)))
model.add(Dropout(0.5))

model.add(Dense(512, 512, init='glorot_uniform'))
model.add(PReLU((512,)))
model.add(BatchNormalization((512,)))
model.add(Dropout(0.5))

model.add(Dense(512, 512, init='glorot_uniform'))
model.add(PReLU((512,)))
model.add(BatchNormalization((512,)))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam")

print("Training model...")

model.fit(X, y, nb_epoch=19, batch_size=16, validation_split=0)

print("Generating submission...")

eva_nn = model.predict_proba(vd)
f = open('eva_nn_19.csv','w')
for l in eva_nn:
    for i in xrange(8):
        f.write(str(l[i])+',')
    f.write(str(l[8])+'\n')
f.close()
op.result = model.predict_proba(X_test)
op.write_result("submission_nn_19_subtrain_0510.csv")

X_train = pd.DataFrame(op.train_features).astype(np.float32)
train_labels = pd.DataFrame(op.train_classes)

test2 = pd.DataFrame(op.test).astype(np.float32)
X_train, scaler2 = preprocess_data(X_train)
y2, encoder2 = preprocess_labels(train_labels)

X_test2, _ = preprocess_data(test2, scaler2)
model.fit(X_train, y2, nb_epoch=19, batch_size=16, validation_split=0)
op.result=model.predict_proba(X_test2)
op.write_result("submission_nn_19_full_0510.csv")

