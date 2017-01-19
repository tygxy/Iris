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
