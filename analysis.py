# coding:utf-8
import pandas as pd 
import seaborn as sns
sns.set(style='white',color_codes=True)
import numpy as np
from sklearn import svm,neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


###  导入数据
iris_data = pd.read_csv("data/Iris.csv")

###  数据可视化

# 直方图
# sns.distplot(a=iris_data['SepalLengthCm'])

# KDE图
#sns.FacetGrid(iris_data, hue="Species", size=10).map(sns.kdeplot,"SepalLengthCm").add_legend()

# 散点图
# sns.FacetGrid(iris_data, hue="Species",size=10).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()

# boxplot
# sns.boxplot(x="Species",y="PetalLengthCm",data=iris_data)

# 两两求散点图
# sns.pairplot(iris_data.drop("Id", axis=1), hue="Species", size=2)
# sns.plt.show()

###  数据清洗

iris_data = iris_data.drop('Id',axis=1)
iris_data.loc[iris_data['Species'] == 'Iris-setosa','Species'] = 0
iris_data.loc[iris_data['Species'] == 'Iris-versicolor','Species'] = 1
iris_data.loc[iris_data['Species'] == 'Iris-virginica','Species'] = 2

iris_data['Species'] = iris_data['Species'].astype('int')

iris_train,iris_test = train_test_split(iris_data,test_size=0.8,random_state=1)

iris_train_y = iris_train['Species']
iris_train_x = iris_train.drop(['Species','SepalLengthCm','SepalWidthCm'],axis=1)


iris_test_y = iris_test['Species']
iris_test_x = iris_test.drop(['Species','SepalLengthCm','SepalWidthCm'],axis=1)


### 机器学习过程

# LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(iris_train_x,iris_train_y)
# y_pred = logreg.predict(iris_test_x)
# print metrics.accuracy_score(iris_test_y,y_pred)

# KNN
# k_range = range(1,10)
# score_list = []
# for k in k_range:	
# 	knn = KNeighborsClassifier(n_neighbors=k)
# 	knn.fit(iris_train_x,iris_train_y)
# 	y_pred = knn.predict(iris_test_x)
# 	score_list.append(metrics.accuracy_score(iris_test_y,y_pred))

# plt.plot(k_range,score_list)
# plt.title('KNN')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.show()

# SVM
# # svm_classifer = svm.SVC()
# svm_classifer = svm.SVC(kernel='rbf', gamma=0.7)
# # svm_classifer = svm.SVC(kernel='poly', degree=5)
# svm_classifer.fit(iris_train_x,iris_train_y)
# y_pred = svm_classifer.predict(iris_test_x)
# print metrics.accuracy_score(y_pred,iris_test_y)

# Decision Tree
# tree = DecisionTreeClassifier()
# tree.fit(iris_train_x,iris_train_y)
# y_pred = tree.predict(iris_test_x)
# print metrics.accuracy_score(y_pred,iris_test_y)


