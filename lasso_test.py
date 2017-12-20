#coding:utf-8
# 岭回归  L2
# lasso L1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# generate some sparse data to play with
np.random.seed(42)
n_samples,n_features=50,200
X=np.random.randn(n_samples,n_features)
coef=3*np.random.randn(n_features)
inds=np.arange(n_features)
np.random.shuffle(inds)
# sparse
coef[inds[10:]]=0
# 生成y
y=np.dot(X,coef)

# add noise  高斯分布  
y+=0.01*np.random.normal((n_samples,))

# split data in train set and test set
n_samples=X.shape[0]
X_train,y_train=X[:n_samples//2],y[:n_samples//2]
X_test,y_test=X[n_samples//2:],y[n_samples//2:]

from sklearn.linear_model import Lasso  # L1
alpha=0.1
lasso=Lasso(alpha=alpha)

y_pred_lasso=lasso.fit(X_train,y_train).predict(X_test)
# R2 已封装好函数
r2_score_lasso=r2_score(y_test,y_pred_lasso)
print lasso
print 'r2:',r2_score_lasso