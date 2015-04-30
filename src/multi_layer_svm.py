__author__ = 'Zhao'
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from src.classes.otclass import OttoProject
f=open("log.txt","w")
op = OttoProject("../data/")
op.load_original()
train = op.train_features
label = op.train_classes
svr = SVC(probability=True)
param_grid = [{"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma":[0.1, 0.01, 0.001, 0.0001]}]
if __name__ == '__main__':
    label_2v3 = label[(label == "Class_2")+(label == "Class_3")]
    train_2v3 = train[(label == "Class_2")+(label == "Class_3")]
    label_7v8 = label[(label == "Class_7")+(label == "Class_8")]
    train_7v8 = train[(label == "Class_7")+(label == "Class_8")]
    label_1v9 = label[(label == "Class_1")+(label == "Class_9")]
    train_1v9 = train[(label == "Class_1")+(label == "Class_9")]
    cv_2v3 = GridSearchCV(svr, param_grid, n_jobs=-1)
    cv_7v8 = GridSearchCV(svr, param_grid, n_jobs=-1)
    cv_1v9 = GridSearchCV(svr, param_grid, n_jobs=-1)
    cv_2v3.fit(train_2v3, label_2v3)
    cv_7v8.fit(train_7v8, label_7v8)
    cv_1v9.fit(train_1v9, label_1v9)
    model_2v3 = SVC(C=cv_2v3.best_params_["C"], gamma=cv_2v3.best_params_["gamma"], probability=True)
    model_7v8 = SVC(C=cv_7v8.best_params_["C"], gamma=cv_7v8.best_params_["gamma"], probability=True)
    model_1v9 = SVC(C=cv_1v9.best_params_["C"], gamma=cv_1v9.best_params_["gamma"], probability=True)

    model_5 = SVC(C=10, gamma=0.01, probability=True)
    label_5 = (label == "Class_5")
    model_5.fit(train, label_5)
    model_4 = SVC(C=10, gamma=0.01, probability=True)
    label_4 = (label == "Class_4")
    model_4.fit(train, label_4)
    model_6 = SVC(C=10, gamma=0.001, probability=True)
    label_6 = (label == "Class_6")
    model_6.fit(train, label_6)
    model_23 = SVC(C=100, gamma=0.001, probability=True)
    label_23 = (label == "Class_2")+(label == "Class_3")
    model_23.fit(train, label_23)
    model_78 = SVC(C=10, gamma=0.001, probability=True)
    label_78 = (label == "Class_7")+(label == "Class_8")
    model_78.fit(train, label_78)
    model_19 = SVC(C=100, gamma=0.001, probability=True)
    label_19 = (label == "Class_1")+(label == "Class_9")
    model_19.fit(train, label_19)
    N = np.shape(op.test)[0]
    index_5 = list(model_5.classes_).index(True)
    index_4 = list(model_4.classes_).index(True)
    index_6 = list(model_6.classes_).index(True)
    index_23 = list(model_23.classes_).index(True)
    index_78 = list(model_78.classes_).index(True)
    index_19 = list(model_19.classes_).index(True)
    index_2 = list(model_2v3.classes_).index("Class_2")
    index_7 = list(model_7v8.classes_).index("Class_7")
    index_1 = list(model_1v9.classes_).index("Class_1")


    for i in xrange(N):
        test = op.test[i]
        prob = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred_5 = model_5.predict_proba(test)
        prob[4] = pred_5[0][index_5]
        rest = float(pred_5[0][1-index_5])
        pred_19 = model_19.predict_proba(test)
        pred_1v9 = model_1v9.predict_proba(test)
        prob[0] = rest*pred_19[0][index_19]*pred_1v9[0][index_1]
        prob[8] = rest*pred_19[0][index_19]*pred_1v9[0][1-index_1]
        rest *= float(pred_19[0][1-index_19])
        pred_6 = model_6.predict_proba(test)
        prob[5] = rest*pred_6[0][index_6]
        rest *= pred_6[0][1-index_6]
        pred_4 = model_4.predict_proba(test)
        prob[3] = rest*pred_4[0][index_4]
        rest *= pred_4[0][1-index_4]
        pred_23 = model_23.predict_proba(test)
        pred_2 = model_2v3.predict_proba(test)
        prob[1] = rest*pred_23[0][index_23]*pred_2[0][index_2]
        prob[2] = rest*pred_23[0][index_23]*pred_2[0][1-index_2]
        rest *= pred_23[0][1-index_23]
        pred_7 = model_7v8.predict_proba(test)
        prob[6] = rest*pred_7[0][index_7]
        prob[7] = rest*pred_7[0][1-index_7]
        op.result.append(prob)
    op.write_result("submission_msvm.csv")



