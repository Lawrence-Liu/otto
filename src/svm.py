__author__ = 'Zhao'
from numpy import *
from sklearn import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import random
random.seed(111)
read_feature = []
read_class = []
header = []
with open("train.csv", "r") as f:
    header = f.readline().strip().split(",")
    for l in f.readlines():
        l = l.strip().split(",")
        l.pop(0)  # remove the id from feature
        read_class.append(l.pop(93))
        l = map(float, l)
        read_feature.append(l)
read_feature = array(read_feature)
read_class = array(read_class)
sss = cross_validation.StratifiedShuffleSplit(read_class, n_iter=1, test_size=0.2)
for train_index, test_index in sss:
    train_feature = read_feature[train_index]
    train_class = read_class[train_index]
    test_feature = read_feature[test_index]
    test_class = read_class[test_index]

# use pca as feature extraction
pca40 = PCA(40)
pca60 = PCA(60)
pca93 = PCA(93)
pca40.fit(train_feature, train_class)
pca60.fit(train_feature, train_class)
pca93.fit(train_feature, train_class)
train_feature40 = pca40.transform(train_feature)
train_feature60 = pca60.transform(train_feature)
train_feature93 = pca93.transform(train_feature)
svm = SVC(probability=True)
parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}
clf = GridSearchCV(svm, parameters)

#clf.fit(train_feature93,train_class)

best_c = SVC(kernel='rbf', C=10, probability=True)
best_c.fit(train_feature93, train_class)

with open("test.csv","r") as f:
    test_list = []
    test_head = f.readline()
    for x in xrange(144368):
        l = f.readline()
        l = l.strip().split(',')
        del l[0]
        l = map(float, l)
        test_list.append(l)

test_pca = pca93.transform(test_list)
prob_list = best_c.predict_proba(test_pca)
with open("submit.csv","w") as f:
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
    for i in xrange(144368):
        f.write(str(i+1)+','+str(prob_list[i][0])+','+str(prob_list[i][1])+','+str(prob_list[i][2])+','+str(prob_list[i][3])+','+str(prob_list[i][4])+','+str(prob_list[i][5])+','+str(prob_list[i][6])+','+str(prob_list[i][7])+','+str(prob_list[i][8])+'\n')
