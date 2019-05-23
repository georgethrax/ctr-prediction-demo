
项目地址：[https://github.com/georgethrax/ctr-prediction-demo](https://github.com/georgethrax/ctr-prediction-demo)

# 环境配置

## 安装Anaconda3：python3发行版
安装Anaconda3后，本文所用到的python库就已经包含在内了。
从 https://www.anaconda.com/distribution/ 下载安装包并安装即可。

## 运行代码
从GitHub下载本项目https://github.com/georgethrax/ctr-prediction-demo 后，有以下几种方式来运行代码：

### 通过jupyter notebook运行代码（推荐的方式）
#### 打开控制台（Windows CMD，Linux/MacOS Terminal），跳转到本项目文件所在的目录
```
cd ctr-prediction-demo
```
#### 启动jupyter notebook
```
jupyter notebook
```
此时会自动浏览器
#### 打开本项目中的`ctr-prediction-demo.ipynb`文件，按顺序执行代码即可

### 通过spyder运行代码
spyder是随Anaconda安装好的一个轻量级python IDE。用spyder打开`ctr_prediction-demo.py`并运行即可。

### 直接在控制台运行代码
#### 打开控制台（Windows CMD，Linux/MacOS Terminal），跳转到本项目文件所在的目录
```
cd ctr-prediction-demo
```
#### 执行代码
```
python ctr-prediction-demo.py
```

# 问题描述
- 问题背景：2015在线广告点击率(CTR)预估大赛 https://www.kaggle.com/c/avazu-ctr-prediction

- 任务目标：根据广告的特征数据，预测一个广告是否被用户点击(点击/未点击的二分类问题)

- 数据文件：`ctr_data.csv`。原始数据过大，这里截取10000条数据。

- 数据字段：
    - id 
    - click 是否点击，0/1 
    - hour 
    - C1 一个个类别型特征(categorical feature)，具体业务含义被隐去
    - banner_pos
    - site_id
    - site_domain
    - site_category
    - app_id
    - app_domain
    - app_category
    - device_id
    - device_ip
    - device_model
    - device_type
    - device_conn_type
    - C14-C21 一些类别型特征

其中，id不使用，click 被作为标签，其他字段可以被用作特征

# 收集数据

这里假设数据已经收集并整理为磁盘文件ctr_data.csv


```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import warnings
warnings.simplefilter("ignore")
```


```python
# 读取数据集
df = pd.read_csv("./ctr_data.csv", index_col=None)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>click</th>
      <th>hour</th>
      <th>C1</th>
      <th>banner_pos</th>
      <th>site_id</th>
      <th>site_domain</th>
      <th>site_category</th>
      <th>app_id</th>
      <th>app_domain</th>
      <th>...</th>
      <th>device_type</th>
      <th>device_conn_type</th>
      <th>C14</th>
      <th>C15</th>
      <th>C16</th>
      <th>C17</th>
      <th>C18</th>
      <th>C19</th>
      <th>C20</th>
      <th>C21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000009e+18</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>15706</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000017e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15704</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000037e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15704</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000064e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15706</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000068e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>fe8cc448</td>
      <td>9166c161</td>
      <td>0569f928</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>18993</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>157</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



# 特征工程
为简单起见，这里仅考虑特征选择和类别型特征编码。

实际场景中，可能面临缺失值处理、离群点处理、日期型特征编码、数据降维等等。

## 特征选择
设置用到的字段/特征/列


```python
cols_data = ['C1','banner_pos', 'site_domain',  'site_id', 'site_category','app_id', 'app_category',  'device_type',  'device_conn_type', 'C14', 'C15','C16']
cols_label = ['click']
```

由设置好的特征字段，构造数据集X和标签y


```python
X = df[cols_data] 
y = df[cols_label]  
```

## 特征编码
特征编码：将原始数据的字符串等特征转换为模型能够处理的数值型特征。LR,SVM类模型可以使用OneHotEncoder。决策树类模型可以使用LabelEncoder。

为简单起见，本文仅讨论决策树类模型，故仅使用LabelEncoder特征编码


```python
cols_categorical = ['site_domain', 'site_id', 'site_category', 'app_id', 'app_category']
lbl = LabelEncoder()

for col in cols_categorical:
    print(col)
    X[col] = lbl.fit_transform(X[col])

```

    site_domain
    site_id
    site_category
    app_id
    app_category



```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C1</th>
      <th>banner_pos</th>
      <th>site_domain</th>
      <th>site_id</th>
      <th>site_category</th>
      <th>app_id</th>
      <th>app_category</th>
      <th>device_type</th>
      <th>device_conn_type</th>
      <th>C14</th>
      <th>C15</th>
      <th>C16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>15706</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15704</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15704</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15706</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>1</td>
      <td>169</td>
      <td>374</td>
      <td>0</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18993</td>
      <td>320</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



## 划分训练集、测试集
这里采用训练集占80%，测试集占20%


```python
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

```

# 建立一个模型，并训练、测试

这里调用一个sklearn算法库中现成的决策树分类器DecisionTreeClassifier，记为clf1

## 创建模型
创建一个分类模型，命名为clf1，使用默认模型参数


```python
clf1 = DecisionTreeClassifier()
```

## 训练
在训练集上训练分类器clf1


```python
clf1.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



## 预测
使用训练好的分类器clf1，在测试集上预测分类结果
预测结果有两种形式
- y_score: 为每个测试样本 x 预测一个0.0~1.0的实数，表示 x 被分类为类别1的概率
- y_pred: 为每个测试样本 x 预测一个0/1类别标签。当 y_score(x) > 0.5 时，y_pred(x) = 1。当 y_score(x) < 0.5 时，y_pred(x) = 0。


```python
y_score = clf1.predict_proba(X_test)[:, clf1.classes_ == 1] #分类器预测的类别为1的概率值/分数值
y_pred = clf1.predict(X_test)        #按阈值(默认0.5)将y_score二值化为0/1预测标签
```

## 评估
评估预测结果，使用ACC, AUC, logloss等评价指标。ACC, AUC越接近于1，logloss越小，分类效果越好。


```python
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)
```

    0.818 0.6621736672845748 2.0666912551211327


# 修改模型参数，重新训练、测试

为模型clf1换一组参数，记为clf1_p1

出于演示目的，不妨令clf1_p1中的一个模型参数修改为 max_leaf_nodes=10。(clf1原参数为max_leaf_nodes=None)

## 创建模型


```python
clf1_p1 = DecisionTreeClassifier( max_leaf_nodes=10)
```

## 训练


```python
clf1_p1.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=10,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



## 预测


```python
y_score = clf1_p1.predict_proba(X_test)[:, clf1_p1.classes_ == 1] 
y_pred = clf1_p1.predict(X_test) 
```

## 评估


```python
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)
```

    0.828 0.6583099862375015 0.43256841278416175


**从评估指标来看，模型clf1_p1比clf1差。**

# 更换模型，重新训练、测试

这里换一个sklearn库中现成的GradientBoostingClassifier，记为clf2


```python
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier() 

```

## 训练


```python
clf2.fit(X_train, y_train) 
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=None, subsample=1.0, verbose=0,
                  warm_start=False)



## 预测


```python
y_score = clf2.predict_proba(X_test)[:, clf2.classes_ == 1] 
y_pred = clf2.predict(X_test) 
```

## 评估


```python
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)
```

    0.8225 0.6870040936411639 0.4252623099250592


**从测试集上的评估指标来看，模型clf2比clf1,clf1_p1好**

# 模型迭代
将收集数据、特征工程、模型选择、模型参数选择、训练测试等步骤反复迭代，直到评价指标令人满意为止。
