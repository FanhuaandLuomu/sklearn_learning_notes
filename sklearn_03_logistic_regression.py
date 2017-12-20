#coding:utf-8
# sklearn 逻辑斯蒂回归
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

digits=datasets.load_digits()

x,y=digits.data,digits.target
x/=255.0
print x
print x.shape
print y
print y.shape

# 计算均值和方差  特征缩放
x=StandardScaler().fit_transform(x)

# classify small aganist large digits
# 转化为两分类  大于4 和小于等于4
y=(y>4).astype(np.int)
print y   # [0,0,...,1,1]

# set regularization parameter
# C 默认1.0  100太大 正则项不起作用,相当于未加正则项  
# 0.01太小,正则项过分起作用  权值稀疏（衰减）的厉害
for i,C in enumerate([100,1,0.01]):
	# turn down tolerance for short training time
	clf_l1_lr=LogisticRegression(C=C,penalty='l1',tol=0.01)
	clf_l2_lr=LogisticRegression(C=C,penalty='l2',tol=0.01)
	clf_l1_lr.fit(x,y)
	clf_l2_lr.fit(x,y)

	# l1 权值稀疏
	coef_l1_lr=clf_l1_lr.coef_.ravel()
	print coef_l1_lr.shape
	coef_l2_lr=clf_l2_lr.coef_.ravel()
	print coef_l2_lr.shape

	# coef_l1_lr contains zeros due to l1 sparsity inducing norm

	sparsity_l1_lr=np.mean(coef_l1_lr==0)*100
	sparsity_l2_lr=np.mean(coef_l2_lr==0)*100

	print 'C=%.2f' %C
	print 'Sparsity with L1 penalty:%.2f%%' %(sparsity_l1_lr)
	print 'score with L1 penalty:%.2f%%' %(clf_l1_lr.score(x,y))

	print 'Sparsity with L2 penalty:%.2f%%' %(sparsity_l2_lr)
	print 'score with L2 penalty:%.2f%%' %(clf_l2_lr.score(x,y))

	l1_plot=plt.subplot(3,2,2*i+1)
	l2_plot=plt.subplot(3,2,2*(i+1))

	if i==0:
		l1_plot.set_title('L1 penalty')
		l2_plot.set_title('L2 penalty')

	l1_plot.imshow(np.abs(coef_l1_lr.reshape(8,8)),interpolation='nearest',
			cmap='binary',vmax=1,vmin=0)
	l2_plot.imshow(np.abs(coef_l2_lr.reshape(8,8)),interpolation='nearest',
			cmap='binary',vmax=1,vmin=0)

	plt.text(-8,3,'C=%.2f' %C)

	l1_plot.set_xticks(())
	l1_plot.set_yticks(())
	l2_plot.set_xticks(())
	l2_plot.set_yticks(())

plt.show()

# multinomial and one-vs-rest logistic regression
from sklearn.datasets import make_blobs

# make 3-classes datases for classification
centers=[[-5,0],[0,1.5],[5,-1],[0,5]]
x,y=make_blobs(n_samples=1000,centers=centers,random_state=40)
transformation=[[0.4,0.2],[-0.4,1.2]]
x=np.dot(x,transformation)

# 多分类  multinomial 性能优于 one vs rest
for multi_class in ['multinomial','ovr']:
	clf=LogisticRegression(solver='sag',max_iter=50,random_state=42,
				multi_class=multi_class).fit(x,y)
	print 'training score:%.3f(%s)' %(clf.score(x,y),multi_class)

for i,color in zip([0,1,2,3],'brgy'):
	idx=np.where(y==i)
	plt.scatter(x[idx,0],x[idx,1],c=color,cmap=plt.cm.Paired,edgecolor='black',s=20)

plt.show()	



