__author__ = 'Zhao'
from src.classes.otclass import OttoProject
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def logloss(act, pred):
    epsilon = 10 ** -15
    pred = np.maximum(np.minimum(pred, 1 - epsilon), epsilon)
    lb = LabelBinarizer()
    lb.fit(act)
    act_binary = lb.transform(act)
    logloss = - np.sum(np.multiply(act_binary, np.log(pred))) / pred.shape[0]
    return logloss

op = OttoProject('../data/')
op.load_original()
vd_labels = op.evaluate_classes
fout1 = open('../bagging/log1.txt','w')
fout2 = open('../bagging/log2.txt','w')
combine = np.zeros((12377, 9))
for i in xrange(30):
    name = 'eva_svm_'+str(i)+'.csv'
    fin = open('../bagging/'+name,'r')
    eva_svm = []
    fin.readline()
    for line in fin.readlines():
        line = line.strip().split(',')
        del line[0]
        line = map(float,line)
        eva_svm.append(line)
    eva_svm = np.array(eva_svm)
    ll = logloss(vd_labels,eva_svm)
    fout1.write("log-loss of bag_"+str(i)+" is "+str(ll)+'\n')
    combine = combine*float(i)/float(i+1)+eva_svm*1.0/float(i+1)
    combine_ll = logloss(vd_labels,combine)
    fout2.write("combine log loss of bag_0 to bay_"+str(i)+" is "+str(combine_ll)+'\n')
fout1.close()
fout2.close()
