__author__ = 'Zhao'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cross_validation import StratifiedShuffleSplit
class OttoProject:
    def __init__(self, path=""):
        self.PATH = path  # the path to parent directory saving test.csv and train.csv, such as "../data/", or "~/"
        self.train_header = [] # header of train.csv
        self.train_features = []  # feature set of train.csv
        self.train_classes = []  # class set of train.csv
        self.evaluate_features = []  # evaluate set(20%) of train.csv used for model evaluating
        self.evaluate_classes = []
        self.subtrain_features = []  # subset(80%) of train.csv used for model training
        self.subtrain_classes = []
        self.test = []  # test.csv
        self.result = []  # saving the prediction result (probability)
        self.pca_train_features = []
        self.pca_test = []
        self.pca_explained_variance = []
        self.lda_train_features = []
        self.lda_test = []
        self.lda_coef = []

    def load_csv(self):
        train_list = []
        with open(self.PATH+"train.csv", "r") as f:
            self.train_header = f.readline().strip().split(",")
            for line in f.readlines():
                line = line.strip().split(',')
                del line[0]
                self.train_classes.append(line.pop(93))
                line = map(float, line)
                self.train_features.append(line)
        with open(self.PATH+"test.csv", "r") as f:
            test_header = f.readline()
            for line in f.readlines():
                line = line.strip().split(',')
                del line[0]
                line = map(float, line)
                self.test.append(line)

    def trans_to_array(self):
        self.train_features = np.array(self.train_features)
        self.test = np.array(self.test)
        self.train_classes = np.array(self.train_classes)

    def write_result(name, self):
        with open(name, "w") as f:
            f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
            for i in xrange(144368):
                f.write(str(i+1)+','+str(self.result[i][0])+','+str(self.result[i][1])+','
                + str(self.result[i][2])+','+str(self.result[i][3])+','+str(self.result[i][4])
                 + ','+str(self.result[i][5])+','+str(self.result[i][6])+','+str(self.result[i][7])
                 + ','+str(self.result[i][8])+'\n')

    def pca(self, n):
        model = PCA(n_components=n)
        model.fit(self.train_features)
        self.pca_train_features = model.transform(self.train_features)
        self.pca_test = model.transform(self.test)
        self.pca_explained_variance = model.explained_variance_ratio_

    def lda(self, n=None):
        model = LDA(solver="eigen", n_components=n)
        model.fit(X=self.train_features, y=self.train_classes)
        self.lda_train_features = model.transform(X=self.lda_train_features)
        self.lda_test = model.transform(self.test)
        self.lda_coef = model.coef_

    def stratified_shuffle(self, train_features): # train_features must be array like
        sss = StratifiedShuffleSplit(self.train_classes, n_iter=1, test_size=0.2)
        for train_index, test_index in sss:
            self.subtrain_features = train_features[train_index]
            self.subtrain_classes = self.train_classes[train_index]
            self.evaluate_features = train_features[test_index]
            self.evaluate_classes = self.train_classes[test_index]

    def load_original(self):
        self.load_csv()
        self.trans_to_array()
        self.stratified_shuffle( self.train_features)
        return self.subtrain_features, self.subtrain_classes, self.evaluate_features, self.evaluate_classes

    def load_pca(self, n=None):
        self.load_csv()
        self.trans_to_array()
        self.pca(n)
        self.stratified_shuffle(self.pca_train_features)
        return self.subtrain_features, self.subtrain_classes, self.evaluate_features, self.evaluate_classes

    def load_lda(self, n=None):
        self.load_csv()
        self.trans_to_array()
        self.lda(n)
        self.stratified_shuffle(self.lda_train_features)
        return self.subtrain_features, self.subtrain_classes, self.evaluate_features, self.evaluate_classes

if __name__ == "__main__":
    op = OttoProject("../../data/")
    op.load_original()  # load untransformed data