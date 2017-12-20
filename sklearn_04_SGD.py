#coding:utf-8
# 随机梯度下降
# SGD Classifier
from sklearn.linear_model import SGDClassifier
x=[[0,0],[1,1]]
y=[0,1]
# penalty:l1 l2 elasticnet (convex combination of l2 and l1)
# elasticnet: (1-l1_ratio)*l2+l1_ratio*l1  mixed
# default:l2
clf=SGDClassifier(loss='hinge',penalty='l1')
clf.fit(x,y)

print clf.predict([[2,2]])
print clf.coef_
print clf.intercept_
print clf.decision_function([[2,2]])

# loss:hinge modified_huber log  
# using loss='log'   enables the 'predict_proba'
clf=SGDClassifier(loss='log').fit(x,y)
# [p0,p1]
print clf.predict_proba([[1,1]])

# SGD:Maximum margin separating hyperplane
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

# create 50 separable points
x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)
print x.shape
print y

# fit the model
clf=SGDClassifier(loss='hinge',alpha=0.01,fit_intercept=True)
clf.fit(x,y)

# plot the line ,the points, and the nearest vectors to the plane
# -1~5 10个等差数列
xx=np.linspace(-1,5,10)
yy=np.linspace(-1,5,10)

x1,x2=np.meshgrid(xx,yy)
print x1.shape,x2.shape
z=np.empty(x1.shape)
# z2=np.empty(x1.shape)

for (i,j),val in np.ndenumerate(x1):
	x_1=val
	
	x_2=x2[i,j]

	print (i,j),x_1,x_2

	p=clf.decision_function([[x_1,x_2]])
	z[i][j]=p[0]

# print '='*10

# for j,y_ in enumerate(yy):
# 	for i,x_ in enumerate(xx):
# 		print (i,j),x_,y_
# 		z2[i][j]=clf.decision_function([[x_,y_]])[0]

# print z==z2

levels=[-1.0,0.0,1.0]
linestyles=['dashed','solid','dashed']
colors='k'

plt.contour(x1,x2,z,levels,colors=colors,linestyles=linestyles)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,edgecolor='black',s=20)

plt.axis('tight')
plt.show()

# weighted samples
# create 20 points
from sklearn import linear_model

np.random.seed(0)
x=np.r_[np.random.randn(10,2)+[1,1],np.random.randn(10,2)]

y=[1]*10+[0]*10

# sample weights
sample_weight=100*np.abs(np.random.randn(20))
# assign a bigger weight to the last 10 samples
idx=np.random.choice(range(20),10,replace=False)
sample_weight[idx]*=10

plt.scatter(x[:, 0], x[:, 1], c=y, s=sample_weight, alpha=0.9,
            cmap=plt.cm.bone, edgecolor='black')

plt.show()

clf=linear_model.SGDClassifier(alpha=0.01)
clf.fit(x,y)
print clf.score(x,y)

clf2=linear_model.SGDClassifier(alpha=0.01)
clf2.fit(x,y,sample_weight=sample_weight)
print clf2.score(x,y)

# comparing various online solvers
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,Perceptron
from sklearn.linear_model import LogisticRegression

heldout=[0.95,0.90,0.75,0.50,0.01]
rounds=20
digits=datasets.load_digits()
x,y=digits.data,digits.target
print x.shape  # (1797,64)
print y.shape

# 比较几种分类器的性能 SAG best
classifiers=[
	('SGD',SGDClassifier()),
	('ASGD',SGDClassifier(average=True)),
	('Perceptron',Perceptron()),
	('SAG',LogisticRegression(solver='sag',tol=1e-1,C=1.e4/x.shape[0]))
]

# train_size
xx=1.-np.array(heldout)

for name,clf in classifiers:
	print 'training %s' %(name)
	rng=np.random.RandomState(42)
	yy=[]
	# test_size
	for i in heldout:
		yy_=[]
		for r in range(rounds):
			x_train,x_test,y_train,y_test=\
					train_test_split(x,y,test_size=i,random_state=rng)
			# fit
			clf.fit(x_train,y_train)
			# predict
			y_pred=clf.predict(x_test)
			# error rate
			yy_.append(1-np.mean(y_pred==y_test))
		yy.append(np.mean(yy_))
	# train_size--error_rate
	plt.plot(xx,yy,label=name)

plt.legend(loc='upper right')
plt.xlabel('train_size')
plt.ylabel('test error rate')
plt.show()

# SVM:For unbalanced data
from sklearn import svm
from sklearn.metrics import recall_score,roc_auc_score,accuracy_score
# we create clusters with 1000 and 100 points
rng=np.random.RandomState(0)
n_samples_1=1000
n_samples_2=100
# 0类范围扩大1.5倍  1 类范围缩小0.5，不过中心位移至2,2
x=np.r_[1.5*rng.randn(n_samples_1,2),0.5*rng.randn(n_samples_2,2)+[2,2]]
y=[0]*n_samples_1+[1]*n_samples_2

# fit the model and get the separating hyperplane
clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(x,y)
print 'SVC(no class_weights):',clf.score(x,y)  # 0.941
pred=clf.predict(x)
# [0.976,0.59]  0.783
print 'SVC(no class_weights):',accuracy_score(y,pred),\
		recall_score(y,pred,average=None),roc_auc_score(y,pred)

# fit the model and get the separating hyperplane using weighted classes 
# set the parameter C of class i to class_weights[i]*C
# 对label为1的类别的样本惩罚项C扩大10倍  默认C=1.0
wclf=svm.SVC(kernel='linear',class_weight={1:10})
wclf.fit(x,y)
print 'SVC(add class_weights):',wclf.score(x,y)  # 0.91
pred=wclf.predict(x)
# [0.901,0.97] 0.937  添加weights， recall都高
print 'SVC(add class_weights):',accuracy_score(y,pred),\
		recall_score(y,pred,average=None),roc_auc_score(y,pred)





