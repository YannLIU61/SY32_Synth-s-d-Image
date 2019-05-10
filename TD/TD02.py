#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:35:02 2019

@author: yann
"""

import numpy as np
import matplotlib.pyplot as plt

def indice_gini(C, probability):
    sum = 0
    for i in range(0, len(C)):
        sum += probability[i] * (1 - probability[i])
    return sum

points = [(1,6), (2,3), (3,5), (5,4), (0,1), (4,2), (6,0), (7,7)]
classe = [-1, -1, -1, -1, 1, 1, 1, 1]
if __name__ == "__main__":
    t_list = list([0, 0])
    max_indice_gini = [-1, -1]
    for i in range(0, len(points[0])):
        # 2 dimensions
        # max_indice_gini = -1
        max_t = -1
        for j in range(0, len(points)):
            # Init t
            t = points[j][i]
            
            D1 = {}
            D2 = {}
            for k in range(0, len(points)):
                if points[k][i] <= t:
                    D1.update({points[k]: classe[k]})
                else:
                    D2.update({points[k]: classe[k]})
            
            g_i = indice_gini([-1, 1], [0.5, 0.5])

            D1_POS = []
            D1_NEG = []
            for d in D1.keys():
                if D1[d] == -1:
                    D1_NEG.append(d)
                else:
                    D1_POS.append(d)

            cl = []; pb = []
            if len(D1_NEG) > 0:
                cl.append(-1)
                pb.append(len(D1_NEG) / len(D1))
            if len(D1_POS) > 0:
                cl.append(1)
                pb.append(len(D1_POS) / len(D1))            
            g_d1 = indice_gini(cl, pb)
            D2_POS = []
            D2_NEG = []
            for d in D2.keys():
                if D2[d] == -1:
                    D2_NEG.append(d)
                else:
                    D2_POS.append(d)
            cl = []; pb = []
            if len(D2_NEG) > 0:
                cl.append(-1)
                pb.append(len(D2_NEG) / len(D2))
            if len(D2_POS) > 0:
                cl.append(1)
                pb.append(len(D2_POS) / len(D2)) 
            g_d2 = indice_gini(cl, pb)

            # CART, I(D) - ... 
            g = g_i - len(D1) / (len(D1) + len(D2)) * g_d1 - len(D2) / (len(D1) + len(D2)) * g_d2 


            # print("Point{} {} Dim, t {}, g {}, D1 size {}, D2 size {}".format(points[j], i, t, g, len(D1), len(D2)))

            if g > max_indice_gini[i]:
                max_indice_gini[i] = g
                t_list[i] = t
                # print(t_list)
        
    print(t_list)
    plt.plot([i[0] for i in points[0:4]], [i[1] for i in points[0:4]], 'ro', [i[0] for i in points[4:8]], [i[1] for i in points[4:8]], 'bo',
        [t_list[0], t_list[0], -2], [15, t_list[1], t_list[1]], 'g-')
    plt.show()
    
    
    
    
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

 
#datArr,labelArr = loadDataSet('horseColicTraining2.txt')  # 这里使用了上面获取数据的函数得到数据集和目标变量
datArr = points
labelArr = classe
clf = AdaBoostClassifier(n_estimators=10)    # 指定10个弱分类器
label =labelArr
scores = cross_val_score(clf,np.mat(datArr),label) # 模型 数据集 目标变量
print(scores.mean())



from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(datArr, labelArr)

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names="F",  
                     class_names="C",  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
