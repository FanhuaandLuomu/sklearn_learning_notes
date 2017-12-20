#coding:utf-8
# 使用scipy的sparse matrix进行文本分类
# 其中使用one-hot的tf-idf空间模型对文本进行特征表示
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from scipy import sparse

# 4分类
categories=[
		'alt.atheism',
		'talk.religion.misc',
		'comp.graphics',
		'sci.space'
]

remove=('headers','footers','quotes')
data_train=fetch_20newsgroups(subset='train',categories=categories,
			shuffle=True,random_state=42,remove=remove)
data_test=fetch_20newsgroups(subset='test',categories=categories,
			shuffle=True,random_state=42,remove=remove)
print 'data loaded'

target_names=data_train.target_names
print target_names

def size_mb(docs):
	return sum(len(s.encode('utf-8')) for s in docs)/1e6

data_train_mb=size_mb(data_train.data)
data_test_mb=size_mb(data_test.data)
print '%d documents - %0.3fMB (traing set)' %(len(data_train.data),data_train_mb)
print '%d documents - %0.3fMB (testing set)' %(len(data_test.data),data_test_mb)

# print data_train.data[0]
# print data_train.target

vectorizer=TfidfVectorizer(stop_words='english')
# sparse matrix
x_train=vectorizer.fit_transform(data_train.data)
print len(vectorizer.vocabulary_)  # 26576 个词特征

print x_train.shape  # (2034,26576)
# print x_train[0]
# for c in  x_train.toarray()[0]:  # 26576维的稀疏矩阵
# 	if c!=0:
# 		print c

x_test=vectorizer.transform(data_test.data)
print x_test.shape   # (1253,26576)

y_train,y_test=data_train.target,data_test.target
print y_train

np.random.seed(1234)

# 分类器
clf1=RandomForestClassifier(n_estimators=200)
clf2=RidgeClassifier()
clf3=Perceptron(n_iter=50)
clf4=KNeighborsClassifier(n_neighbors=20)
clf5=SGDClassifier(alpha=0.0001,n_iter=50,penalty='elasticnet')

# 转化为array
# x_train=x_train.toarray()
# x_test=x_test.toarray()

# scale  mean 0 and variance 1
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

# # array 转为稀疏矩阵 sparse matrix
# x_train=sparse.csr_matrix(x_train)
# x_test=sparse.csr_matrix(x_test)

# 使用sparse matrix 速度更快 结果一样
for clf in [clf1,clf2,clf3,clf4,clf5]:
	clf.fit(x_train,y_train)

	preds=clf.predict(x_test)

	print metrics.accuracy_score(y_test,preds)

