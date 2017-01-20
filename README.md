# Iris

## 目录
- 1.数据导入
- 2.数据预览
- 3.数据可视化
- 4.数据清洗
- 5.机器学习
  - LogisticRegression
  - KNN
  - SVM
  - Decision Tree
- 6.结果可视化

## 1. 数据导入
- 相关库导入
``` python
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
```
- 数据导入
``` python
iris_data = pd.read_csv("data/Iris.csv")
```

## 2. 数据预览
- 数据的基本信息
``` python
> iris_data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
Id               150 non-null int64
SepalLengthCm    150 non-null float64
SepalWidthCm     150 non-null float64
PetalLengthCm    150 non-null float64
PetalWidthCm     150 non-null float64
Species          150 non-null object
dtypes: float64(4), int64(1), object(1)
memory usage: 7.1+ KB
None
```

- 数据共有150个，没有缺省值
- 有五个属性，一个类别
  
``` python
> iris_data.head()

   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
```

 - 五个属性分别为Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm
 
``` python
> iris_data['Species'].value_counts()

Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
```

 - 类型共有三种，分别为Iris-setosa,Iris-versicolor,Iris-virginica

## 3. 数据可视化
- SepalLengthCm属性的直方图
``` python
sns.distplot(a=iris_data['SepalLengthCm'])
sns.plt.show()
```
![](raw/figure_1.png?raw=true)

- SepalLengthCm属性的KDE图（Kernel Density Estimation）
``` python
sns.FacetGrid(iris_data, hue="Species", size=10).map(sns.kdeplot,"SepalLengthCm").add_legend()
sns.plt.show()
```
![](raw/figure_2.png?raw=true)

- SepalLengthCm和SepalWidthCm属性的散点图
``` python
sns.FacetGrid(iris_data, hue="Species",size=10).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
sns.plt.show()
```
![](raw/figure_3.png?raw=true)

- PetalLengthCm属性的boxplot图
``` python
sns.boxplot(x="Species",y="PetalLengthCm",data=iris_data)
sns.plt.show()
```
![](raw/figure_4.png?raw=true)

- 两两属性求散点图
``` python
sns.pairplot(iris_data.drop("Id", axis=1), hue="Species", size=2)
sns.plt.show()
```
![](raw/figure_5.png?raw=true)
  
    可以选择PetalLengthCm和PetalWidthCm两个属性做训练
    
## 4. 数据清洗
``` python
# 去掉Id列
iris_data = iris_data.drop('Id',axis=1)
# Species的值修改成数字，并转化类型成int
iris_data.loc[iris_data['Species'] == 'Iris-setosa','Species'] = 0
iris_data.loc[iris_data['Species'] == 'Iris-versicolor','Species'] = 1
iris_data.loc[iris_data['Species'] == 'Iris-virginica','Species'] = 2
iris_data['Species'] = iris_data['Species'].astype('int')
# 数据集按照8：2分成训练集和测试集合
iris_train,iris_test = train_test_split(iris_data,test_size=0.2,random_state=1)
# 选取PetalLengthCm和PetalWidthCm两个属性做训练
iris_train_y = iris_train['Species']
iris_train_x = iris_train.drop(['Species','SepalLengthCm','SepalWidthCm'],axis=1)
iris_test_y = iris_test['Species']
iris_test_x = iris_test.drop(['Species','SepalLengthCm','SepalWidthCm'],axis=1)
```
## 5. 机器学习
- LogisticRegression
``` python
logreg = LogisticRegression()
logreg.fit(iris_train_x,iris_train_y)
y_pred = logreg.predict(iris_test_x)
print metrics.accuracy_score(iris_test_y,y_pred)
```
    
    准确率为0.7

- KNN
``` python
k_range = range(1,10)
score_list = []
for k in k_range:	
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(iris_train_x,iris_train_y)
	y_pred = knn.predict(iris_test_x)
	score_list.append(metrics.accuracy_score(iris_test_y,y_pred))

plt.plot(k_range,score_list)
plt.title('KNN')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
```

![](raw/figure_7.png?raw=true)

    选取k的值不同，准确率也不同，我们可以看到当k为1和2时，准确率为1.0

- SVM
``` python
# svm_classifer = svm.SVC()
svm_classifer = svm.SVC(kernel='rbf', gamma=0.7)
# svm_classifer = svm.SVC(kernel='poly', degree=5)
svm_classifer.fit(iris_train_x,iris_train_y)
y_pred = svm_classifer.predict(iris_test_x)
accuracy = metrics.accuracy_score(y_pred,iris_test_y)
print  accuracy
```
    
    SVM核函数不同，预测结果也不同。这里选择了高斯核，准确率为0.966666666667

- Decision Tree
``` python
tree = DecisionTreeClassifier()
tree.fit(iris_train_x,iris_train_y)
y_pred = tree.predict(iris_test_x)
print metrics.accuracy_score(y_pred,iris_test_y)
```
    
    准确率为0.966666666667
    

## 6. 结果可视化
- 高斯核SVM模型，可视化表示预测范围
 
 ![](raw/figure_6.png?raw=true)
 
- LogisticRegression模型，可视化表示预测范围
 ![](raw/figure_9.png?raw=true)
 
