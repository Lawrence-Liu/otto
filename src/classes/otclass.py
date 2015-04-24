__author__ = 'Zhao'
import numpy as np


class OttoProject:
    def __init__(self):
        self.PATH = ""
        self.train_header = []
        self.train_features = []
        self.train_classes = []
        self.test = []
        self.result = []

    def load_csv(self):
        train_list=[]
        with open(self.PATH, "r") as f:
            self.train_header = f.readline().strip().split(",")
            for line in f.readlines():
                line = line.strip().split(',')
                del line[0]
                self.train_classes.append(line.pop(93))
                self.train_features.append(line)

    def trans_to_array(self):
        self.train_features = np.array(self.train_features)

    def write_result(name,self):
        with open(name, "w") as f:
            f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
            for i in xrange(144368):
                f.write(str(i+1)+','+str(self.result[i][0])+','+str(self.result[i][1])+','
                + str(self.result[i][2])+','+str(self.result[i][3])+','+str(self.result[i][4])
                 + ','+str(self.result[i][5])+','+str(self.result[i][6])+','+str(self.result[i][7])
                 + ','+str(self.result[i][8])+'\n')
