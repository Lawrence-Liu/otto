__author__ = 'Zhao'
from src.classes.otclass import OttoProject
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import numpy as np
if __name__ == '__main__':
    op = OttoProject("../data/")
    op.load_original()
    range_gamma = list(np.exp2(range(-5, 15)))
    range_C = list(np.exp2(range(-16,9)))
    params = [{"gamma":range_gamma, "C":range_C}]
    svr = SVC()
    clf = GridSearchCV(svr,params,n_jobs=6,verbose=2)
    clf.fit(op.subtrain_features, op.subtrain_classes)
    print str(clf.best_score_)+"\n"
    print str(clf.best_params_)+"\n"
    print str(clf.grid_scores_)+"\n"

