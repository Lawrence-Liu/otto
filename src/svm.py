__author__ = 'Zhao'
from src.classes.otclass import OttoProject
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from logloss import logloss
op = OttoProject('../data/')
op.load_original()
train = op.train_features
train_label = op.train_classes
test = op.test
model = SVC(probability=True)
params = [{'C': np.exp2(range(-5, 15)), 'gamma': np.exp2(range(-16, 9))}]
f = open("log.txt", "w")
if __name__ == "__main__":
    clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=6, cv=5)
    clf.fit(train, train_label)
    op.result = clf.predict_proba(test)
    op.write_result("submission-0503-svm.csv")
    f.write("log-loss on training set:\n")
    f.write(str(logloss(train_label, clf.predict_proba(train))))
    f.write("best params:\n")
    f.write("Gamma:"+str(clf.best_params_["gamma"])+",C:"+str(clf.best_params_["C"])+"\n")



