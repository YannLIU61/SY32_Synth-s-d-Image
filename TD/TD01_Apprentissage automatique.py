#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

d_train = np.loadtxt('SY32_P19_TD01_data.csv')

def err_emp(D, h):
#    sum = 0
#    for item in D:
#        if item[0] <= h and item[1] > 0:
#            sum += 1
#        elif item[0] > h and item[1] < 0:
#            sum += 1
#      err = sum / len(D)

    err = np.sum(np.logical_and(D[:,0] <= h, D[:, 1] > 0)) \
        + np.sum(np.logical_and(D[:,0] > h, D[:, 1] < 0))
    err /= len(D)
    return err

def h_opt(D):
    h_min = np.min(D[:,0])
    h_max = np.max(D[:,0])
    h = h_min
    err_emp_opt = err_emp(D,h_min)
    h_opt = h_min
    while h < h_max:
        if err_emp_opt > err_emp(D,h):
            err_emp_opt = err_emp(D,h)
            h_opt = h
        h+=0.001
    return (h_opt,err_emp_opt)

def h_opt2(D):
    h_iter = np.sort(D[:,0])
    err = [err_emp(D,h) for h in h_iter]
    #np.argmin  最小值下角标
    idx = np.argmin(err)
    return (err[idx],h_iter[idx])

def cv(D):
    # 随机打乱顺序
    np.random.shuffle(D)
    hs = [h_opt2(np.concatenate((D[0:i*60,:],D[(i+1)*60:300,:]))) for i in range (5)][0]
    errs = [err_emp(D[i*60:(i*60+60),:], hs[i]) for i in range (5)]
    return np.mean(errs)
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    d_train[:,0].reshape(-1,1),  d_train[:,1].reshape(-1,1), test_size=0.5, random_state=0)

#在行上合并
D_train = np.hstack((X_train,y_train)) 
#np.vstack((a,b)) 在列上合并
D_test= np.hstack((X_test,y_test))

h_opt_train = h_opt(d_train)[0]
err_emp(D_test,h_opt_train)

# ===================================数据集划分,训练模型==========================
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)  # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
#print('训练集大小：',X_train.shape,y_train.shape)  # 训练集样本大小
#print('测试集大小：',X_test.shape,y_test.shape)  # 测试集样本大小
#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) # 使用训练集训练模型
#print('准确率：',clf.score(X_test, y_test))  # 计算测试集的度量值（准确率）

# ===================================直接调用交叉验证评估模型==========================
#from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)  #cv为迭代次数。
#print(scores)  # 打印输出每次迭代的度量值（准确度）
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）


