#coding:utf-8
#sklearn 决策树
import numpy as np

# 1.分类树
from sklearn import tree
x=[[0,0],[1,1]]
y=[0,1]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)

# 预测类别
res=clf.predict([[2,2]])
print res

# 预测每个类的概率
prob=clf.predict_proba([[0,2]])
print prob

# 多分类 iris数据集
from sklearn.datasets import load_iris
iris=load_iris()
print iris.data.shape
print iris.target.shape

clf=tree.DecisionTreeClassifier()
clf=clf.fit(iris.data,iris.target)

'''
# 可视化二叉树结构
import pydotplus
from IPython.display import Image

dot_data=tree.export_graphviz(clf,out_file='iris.dot',
	feature_names=iris.feature_names,class_names=iris.target_names,
	filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(open('iris.dot').read())

graph.write_pdf('iris_2.pdf')

import os
os.unlink('iris.dot')
'''

# 使用模型进行预测
print clf.predict_proba(iris.data[1]), iris.target[1]
print clf.predict_proba(iris.data[1]), iris.target[1]

# 2.回归树
x=[[0,0],[2,2]]
y=[0.5,2.5]
clf=tree.DecisionTreeRegressor()
clf=clf.fit(x,y)
print clf.predict([[1,2]])

# eg. 
import matplotlib.pyplot as plt

# create a random dataset
rng=np.random.RandomState(1)
x=np.sort(5*rng.rand(80,1),axis=0)  # shape：(80,1)
y=np.sin(x).ravel()  # faltten  (80,)

# y的值 每五个变换
y[::5]+=3*(0.5 - rng.rand(80/5))

# fit regression model
# 设置树的最大深度
regr_1=tree.DecisionTreeRegressor(max_depth=2)
regr_2=tree.DecisionTreeRegressor(max_depth=5)

# fit
regr_1.fit(x,y)
regr_2.fit(x,y)

# [[0.],[0.01],...] 新增一维  等价于 .reshape((-1,1))
x_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]

# predict
y_1=regr_1.predict(x_test)
y_2=regr_2.predict(x_test)

print y_1.shape
print y_2.shape

# plot the resluts
plt.figure()
plt.scatter(x,y,s=20,edgecolor='black',c='darkorange',label='data')
plt.plot(x_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)

plt.plot(x_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)

plt.xlabel('data')
plt.ylabel('target')

plt.title('Decision Tree Regression')

plt.legend()

plt.show()

# 4.multi-output decision tree regression
# 多个输出
# create a random dataset
rng=np.random.RandomState(1)
# 0~1 -> 0~200 ->  -100~100  100个
x=np.sort(200*rng.rand(100,1)-100,axis=0)
print x.shape  # (100,1)
# [[pi*sin(x),pi*cos(x)],...]   两个输出
y=np.array([np.pi*np.sin(x).ravel(), np.pi*np.cos(x).ravel()]).T
print y.shape  # (100,2)

y[::5,:]+=(0.5-rng.rand(20,y.shape[-1]))

# 3个模型  max_depth 不同
regr_1=tree.DecisionTreeRegressor(max_depth=2)
regr_2=tree.DecisionTreeRegressor(max_depth=5)
regr_3=tree.DecisionTreeRegressor(max_depth=8)

# fit
regr_1.fit(x,y)
regr_2.fit(x,y)
regr_3.fit(x,y)

# predict
x_test=np.arange(-100.0,100.0,0.01)[:,np.newaxis]

y_1=regr_1.predict(x_test)
y_2=regr_2.predict(x_test)
y_3=regr_3.predict(x_test)

print y_1.shape
print y_2.shape
print y_3.shape

# plot results
# 可视化  t1^2+t2^2=pi*2  半径为pi的圆
plt.figure()
s1=50
s2=25

plt.scatter(y[:,0],y[:,1],c='navy',s=s1,edgecolor='black',label='data')
plt.scatter(y_1[:,0],y_1[:,1],c='cornflowerblue',s=s1,edgecolor='black',label='max_depth=2')
plt.scatter(y_2[:,0],y_2[:,1],c='red',s=s1,edgecolor='black',label='max_depth=5')
plt.scatter(y_3[:,0],y_3[:,1],c='orange',s=s1,edgecolor='black',label='max_depth=8')

plt.xlim([-6,6])
plt.ylim([-6,6])
plt.xlabel('target 1')
plt.ylabel('target 2')
plt.title('multi-output Decision Tree Regression')
plt.legend(loc='best')
plt.show()

# 5. face completion with a multi-output estimators
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

# load the faces datasets
data=fetch_olivetti_faces()
targets=data.target

data=data.images.reshape((len(data.images),-1))
# (400, 4096)
# train and test
train=data[targets<30]
test=data[targets>=30]

# test on a subset of people
n_faces=5
rng=check_random_state(4)
face_ids=rng.randint(test.shape[0],size=(n_faces,))

test=test[face_ids,:]
print test.shape  # (5,4096)


##
# test
from PIL import Image
im=Image.open('b.jpg')
im=im.convert('L')

d=np.array(im).astype('float32')
d/=255.0

test[0]=d.ravel()

im2=Image.open('c.jpg')
im2=im2.convert('L')

d2=np.array(im2).astype('float32')
d2/=255.0

test[1]=d2.ravel()
# train[2]=d2.ravel()

im2=Image.open('d.jpg')
im2=im2.convert('L')

d2=np.array(im2).astype('float32')
d2/=255.0

test[2]=d2.ravel()

# train[1]=d2.ravel()
# train[0]=d.ravel()
##

# 上部分预测下部分
n_pixels=data.shape[1]  # 4096
# upper half of the faces
x_train=train[:,:n_pixels//2]
# lower half of the faces
y_train=train[:,n_pixels//2:]

x_test=test[:,:n_pixels//2]
y_test=test[:,n_pixels//2:]

# (300,2048) (300,2048)
print x_train.shape,y_train.shape
# (5,2048)  (5,2048)
print x_test.shape,y_test.shape

# fit estimators
ESTIMATORS={
	'Extra trees':ExtraTreesRegressor(n_estimators=10,
		max_features=32,random_state=0),
	'K-nn':KNeighborsRegressor(),
	'Linear regression':LinearRegression(),
	'Ridge':RidgeCV()
}

y_test_predict=dict()
for name, estimator in ESTIMATORS.items():
	estimator.fit(x_train,y_train)
	y_test_predict[name]=estimator.predict(x_test)

# plot the completed faces
image_shape=(64,64)

# 一个原图  4个预测的  5列  5行
n_cols=1+len(ESTIMATORS)
plt.figure(figsize=(2.*n_cols,2.26*n_faces))
plt.suptitle('face completion with multi-output estimators',size=16)

# print y_test_predict

for i in range(n_faces):
	true_face=np.hstack((x_test[i],y_test[i]))
	# i!=0
	if i:
		sub=plt.subplot(n_faces,n_cols,i*n_cols+1)
	else:
		sub=plt.subplot(n_faces,n_cols,i*n_cols+1,title='true faces')

	sub.axis('off')

	sub.imshow(true_face.reshape(image_shape),cmap=plt.cm.gray,
				interpolation='nearest')
	for j,est in enumerate(sorted(ESTIMATORS)):
		completed_face=np.hstack((x_test[i],y_test_predict[est][i]))
		if i:
			sub=plt.subplot(n_faces,n_cols,i*n_cols+2+j)
		else:
			sub=plt.subplot(n_faces,n_cols,i*n_cols+2+j,title=est)

		sub.axis('off')
		sub.imshow(completed_face.reshape(image_shape),
				cmap=plt.cm.gray,interpolation='nearest')

plt.show()

