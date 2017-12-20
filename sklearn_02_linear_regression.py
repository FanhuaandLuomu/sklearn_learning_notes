#coding:utf-8
# sklearn 线性回归

# eg:y=1+x1+2*x2
# w=[1,1,2]  根据x=[[1,0,1],[1,1,1],[1,2,2]]  y=[[3],[4],[7]]拟合w

from sklearn import linear_model
# 默认附加偏置 即 y=w0*x0+w1*x1+w2*x2
reg=linear_model.LinearRegression()
print reg
# 样本中不需给出x0，默认x0=1
reg.fit([[0,1],[1,1],[2,2]],[3,4,7])

# w0  (b)   y=wx+b
print reg.intercept_
# w=[w1,w2,..]
print reg.coef_

# 实例
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

# load the diabetes dataset
diabetes=datasets.load_diabetes()

# use only one feature
# 只使用特征的第二列
# (442,10)->(442,1)
diabetes_x=diabetes.data[:,np.newaxis,2]

# split the data into training/testing sets
diabetes_x_train=diabetes_x[:-20]
diabetes_x_test=diabetes_x[-20:]

# split the target into training/testing sets
diabetes_y_train=diabetes.target[:-20]
diabetes_y_test=diabetes.target[-20:]

# create linear regression object
regr=linear_model.LinearRegression()

# train the model using the training sets
regr.fit(diabetes_x_train,diabetes_y_train)
print regr

# make predictions using the testing sets
diabetes_y_pred=regr.predict(diabetes_x_test)
print diabetes_y_pred.shape

# coefficients
print 'intercept_:',regr.intercept_  # w0
print 'coef_:',regr.coef_  # w1

print 'mean square error:',mean_squared_error(diabetes_y_test,diabetes_y_pred)
print 'variance score:',r2_score(diabetes_y_test,diabetes_y_pred)

# plot 可视化
plt.scatter(diabetes_x_test,diabetes_y_test,color='black')
plt.scatter(diabetes_x_test,diabetes_y_pred,color='red')
plt.plot(diabetes_x_test,diabetes_y_pred,color='blue',linewidth=3)

# plt.xticks((range(1)))
# plt.yticks((range(1)))

plt.show()

# Ridge regression
# L2
reg=linear_model.Ridge(alpha=0.5)
reg.fit([[0,0],[0,0],[1,1]],[0,.1,1])
# w
print reg.coef_
# b
print reg.intercept_

# plot ridge coefficients as a function of the regularization
x=1./(np.arange(1,11)+np.arange(0,10)[:,np.newaxis])  # 10x10
y=np.ones(10)

n_alphas=200
alphas=np.logspace(-10,-1,n_alphas)

coefs=[]
for a in alphas:
	ridge=linear_model.Ridge(alpha=a, fit_intercept=False)
	ridge.fit(x,y)
	coefs.append(ridge.coef_)

# display
# x轴代表alpha,y轴代表weights(10个)的取值，
# 可以看出alpha越大，weights被惩罚的接近0
ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# 正则化参数 alpha 内置交叉验证
reg=linear_model.RidgeCV(alphas=[0.1,1.0,5.0,10.0])
reg.fit([[0,0],[0,0],[1,1]],[0,.1,1])
print reg.predict([[0,0],[0,0],[1,1]])
# r2_score
score=reg.score([[0,0],[0,0],[1,1]],[0,.1,1])
print 'r2_score:',score
print 'reg.alpha_:',reg.alpha_

# Lasso regression  L1
# 权值稀疏
reg=linear_model.Lasso(alpha=0.1)
reg.fit([[0,0],[1,1]],[0,1])
res=reg.predict([[0,0],[1,1]])
print reg.coef_
print reg.intercept_
print res

# Lasso and Elastic Net for Sparse Signals
# Generate some sparse data to play with
np.random.seed(40)

n_samples,n_features=50,200
# 训练样本x 服从高斯分布
x=np.random.randn(n_samples,n_features)
coef=3*np.random.randn(n_features)
inds=np.arange(n_features)
# 随机打乱
np.random.shuffle(inds)
# 部分编号的wi置0  只保留10个w不为0
coef[inds[10:]]=0  # sparsify coef
y=np.dot(x,coef)

# add noise  添加0-mean 1-varis 的噪声
y+=0.01*np.random.normal(size=n_samples)

# split the data in train and test sets
n_samples=x.shape[0]
x_train, y_train=x[:n_samples//2],y[:n_samples//2]
x_test, y_test=x[n_samples//2:],y[n_samples//2:]

# lasso
from sklearn.linear_model import Lasso

alpha=0.1
lasso=Lasso(alpha=alpha)

# fit and predict
y_pred_lasso=lasso.fit(x_train,y_train).predict(x_test)
r2_score_lasso=r2_score(y_test,y_pred_lasso)
print lasso
print 'r2 on test data:%f' %(r2_score_lasso)

# ElasticNet
from sklearn.linear_model import ElasticNet

enet=ElasticNet(alpha=alpha,l1_ratio=0.7)
# fit and predict
y_pred_enet=enet.fit(x_train,y_train).predict(x_test)
r2_score_enet=r2_score(y_test,y_pred_enet)
print enet
print 'r2 on test data:%f' %(r2_score_enet)

# 可视化
# enet 权重
plt.plot(enet.coef_,color='lightgreen',linewidth=2,label='Elastic net coefficients')
# lasso 权重
plt.plot(lasso.coef_,color='gold',linewidth=2,label='lasso coefficients')
# real 权重
plt.plot(coef,'--',color='navy',label='orginal coefficients')

plt.legend(loc='best')
plt.title('Lasso R^2:%f, Elastic Net R^2:%f' %(r2_score_lasso,r2_score_enet))
plt.show()

# Joint feature selection with multi-task Lasso
from sklearn.linear_model import MultiTaskLasso,Lasso

rng=np.random.RandomState(42)
# generate some 2D cofficients with sine waves with random frequency and sparse
n_samples,n_features,n_tasks=100,30,40
n_relevant_features=5
# 40个task 每个task有一组（30,）维的特征权重  (40,30)
coef=np.zeros((n_tasks,n_features))

# (40,)
times=np.linspace(0,2*np.pi,n_tasks)

# coef (40,30)  只有前5列特征权重有效 不为0
for k in range(n_relevant_features):
	coef[:,k]=np.sin((1.+rng.randn(1))*times+3*rng.randn(1))

# (100,30)
x=rng.randn(n_samples,n_features)
# y=x.w.T+bias   (100,40)   100个样本 每个样本有40个任务（sex、age..）
y=np.dot(x,coef.T)+rng.randn(n_samples,n_tasks)

# 1、通过单任务Lasso进行训练  （40,30）
coef_lasso_=np.array([Lasso(alpha=0.5).fit(x,y_).coef_ for y_ in y.T])
print coef_lasso_.shape
print coef_lasso_

# 2、直接通过多任务MultiTaskLasso进行训练  (40,30)  
coef_multi_task_lasso_=MultiTaskLasso(alpha=1.).fit(x,y).coef_
print coef_multi_task_lasso_.shape
print coef_multi_task_lasso_

# plot support and time series
fig=plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.spy(coef_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10,5,'Lasso')

plt.subplot(1,2,2)
plt.spy(coef_multi_task_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or series)')
plt.text(10,5,'MultiTaskLasso')
fig.suptitle('coefficient non-zero location')

# 通过对比0~4 维度的权值 (40个任务，各自取出对应维度的权值，组成一个40维度的权值列表)
# 通过观察曲线，可见MultiTaskLasso的权重拟合效果优于分开拟合的Lasso的权值拟合效果
# (应该是MultiTaskLasso考虑了40个任务的相关性)
for feature_to_plot in range(5):

	# feature_to_plot=4
	plt.figure()
	lw=2
	plt.plot(coef[:,feature_to_plot],color='seagreen',linewidth=lw,label='Ground truth')
	plt.plot(coef_lasso_[:,feature_to_plot],color='cornflowerblue',linewidth=lw,label='lasso')
	plt.plot(coef_multi_task_lasso_[:,feature_to_plot],color='gold',linewidth=lw,label='MultiTaskLasso')

	plt.legend(loc='upper center')
	plt.axis('tight')
	plt.ylim([-1.1,1.1])
	plt.show()

# least angel regression 最小角度回归
from sklearn import datasets

diabetes=datasets.load_diabetes()
x=diabetes.data
print x.shape
y=diabetes.target
print y.shape

print 'computing regularization path using the LARS...'
alphas,_,coefs,iters=linear_model.lars_path(x,y,method='lasso',\
		verbose=True,return_n_iter=True,max_iter=5)
print alphas
print coefs
print iters


xx=np.sum(np.abs(coefs.T),axis=1)
print xx
xx/=xx[-1]
print xx

plt.plot(xx,coefs.T)
ymin,ymax=plt.ylim()
plt.vlines(xx,ymin,ymax,linestyle='dashed')
plt.xlabel('|coef|/max(|coef|)')
plt.ylabel('cofficients')
plt.title('LASSO path')
plt.axis('tight')
plt.show()

# 贝叶斯用于回归
x=[[0.,0.],[1.,1.],[2.,2.],[3.,3.]]
y=[0,1,2,3]

reg=linear_model.BayesianRidge()
reg.fit(x,y)
print reg.coef_

print reg.predict([[2,3]])