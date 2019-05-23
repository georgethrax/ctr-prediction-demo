
# coding: utf-8

# # 环境配置

# ## 安装Anaconda3：python3发行版
# 安装Anaconda3后，本文所用到的python库就已经包含在内了。
# 从 https://www.anaconda.com/distribution/ 下载安装包并安装即可。

# ## 运行代码
# 从GitHub下载本项目https://github.com/georgethrax/ctr-prediction-demo 后，有以下几种方式来运行代码：

# ### 通过jupyter notebook运行代码（推荐的方式）
# #### 打开控制台（Windows CMD，Linux/MacOS Terminal），跳转到本项目文件所在的目录
# ```
# cd ctr-prediction-demo
# ```
# #### 启动jupyter notebook
# ```
# jupyter notebook
# ```
# 此时会自动浏览器
# #### 打开本项目中的`ctr-prediction-demo.ipynb`文件，按顺序执行代码即可

# ### 通过spyder运行代码
# spyder是随Anaconda安装好的一个轻量级python IDE。用spyder打开`ctr_prediction-demo.py`并运行即可。

# ### 直接在控制台运行代码
# #### 打开控制台（Windows CMD，Linux/MacOS Terminal），跳转到本项目文件所在的目录
# ```
# cd ctr-prediction-demo
# ```
# #### 执行代码
# ```
# python ctr-prediction-demo.py
# ```

# # 问题描述
# - 问题背景：2015在线广告点击率(CTR)预估大赛 https://www.kaggle.com/c/avazu-ctr-prediction
# 
# - 任务目标：根据广告的特征数据，预测一个广告是否被用户点击(点击/未点击的二分类问题)
# 
# - 数据文件：`ctr_data.csv`。原始数据过大，这里截取10000条数据。
# 
# - 数据字段：
#     - id 
#     - click 是否点击，0/1 
#     - hour 
#     - C1 一个个类别型特征(categorical feature)，具体业务含义被隐去
#     - banner_pos
#     - site_id
#     - site_domain
#     - site_category
#     - app_id
#     - app_domain
#     - app_category
#     - device_id
#     - device_ip
#     - device_model
#     - device_type
#     - device_conn_type
#     - C14-C21 一些类别型特征
# 
# 其中，id不使用，click 被作为标签，其他字段可以被用作特征

# # 收集数据

# 这里假设数据已经收集并整理为磁盘文件ctr_data.csv

# In[2]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


# In[3]:


# 读取数据集
df = pd.read_csv("./ctr_data.csv", index_col=None)
df


# # 特征工程
# 为简单起见，这里仅考虑特征选择和类别型特征编码。
# 
# 实际场景中，可能面临缺失值处理、离群点处理、日期型特征编码、数据降维等等。

# ## 特征选择
# 设置用到的字段/特征/列

# In[4]:


cols_data = ['C1','banner_pos', 'site_domain',  'site_id', 'site_category','app_id', 'app_category',  'device_type',  'device_conn_type', 'C14', 'C15','C16']
cols_label = ['click']


# 由设置好的特征字段，构造数据集X和标签y

# In[5]:


X = df[cols_data] 
y = df[cols_label]  


# ## 特征编码
# 特征编码：将原始数据的字符串等特征转换为模型能够处理的数值型特征。LR,SVM类模型可以使用OneHotEncoder。决策树类模型可以使用LabelEncoder。
# 
# 为简单起见，本文仅讨论决策树类模型，故仅使用LabelEncoder特征编码

# In[6]:


cols_categorical = ['site_domain', 'site_id', 'site_category', 'app_id', 'app_category']
lbl = LabelEncoder()

for col in cols_categorical:
    print(col)
    X[col] = lbl.fit_transform(X[col])


# In[7]:


X


# ## 划分训练集、测试集
# 这里采用训练集占80%，测试集占20%

# In[8]:


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # 建立一个模型，并训练、测试

# 这里调用一个sklearn算法库中现成的决策树分类器DecisionTreeClassifier，记为clf1

# ## 创建模型
# 创建一个分类模型，命名为clf1，使用默认模型参数

# In[9]:


clf1 = DecisionTreeClassifier()


# ## 训练
# 在训练集上训练分类器clf1

# In[10]:


clf1.fit(X_train, y_train)


# ## 预测
# 使用训练好的分类器clf1，在测试集上预测分类结果
# 预测结果有两种形式
# - y_score: 为每个测试样本 x 预测一个0.0~1.0的实数，表示 x 被分类为类别1的概率
# - y_pred: 为每个测试样本 x 预测一个0/1类别标签。当 y_score(x) > 0.5 时，y_pred(x) = 1。当 y_score(x) < 0.5 时，y_pred(x) = 0。

# In[11]:


y_score = clf1.predict_proba(X_test)[:, clf1.classes_ == 1] #分类器预测的类别为1的概率值/分数值
y_pred = clf1.predict(X_test)        #按阈值(默认0.5)将y_score二值化为0/1预测标签


# ## 评估
# 评估预测结果，使用ACC, AUC, logloss等评价指标。ACC, AUC越接近于1，logloss越小，分类效果越好。

# In[12]:


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)


# # 修改模型参数，重新训练、测试
# 
# 为模型clf1换一组参数，记为clf1_p1
# 
# 出于演示目的，不妨令clf1_p1中的一个模型参数修改为 max_leaf_nodes=10。(clf1原参数为max_leaf_nodes=None)

# ## 创建模型

# In[13]:


clf1_p1 = DecisionTreeClassifier( max_leaf_nodes=10)


# ## 训练

# In[14]:


clf1_p1.fit(X_train, y_train)


# ## 预测

# In[15]:


y_score = clf1_p1.predict_proba(X_test)[:, clf1_p1.classes_ == 1] 
y_pred = clf1_p1.predict(X_test) 


# ## 评估

# In[16]:


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)


# **从评估指标来看，模型clf1_p1比clf1差。**

# # 更换模型，重新训练、测试
# 
# 这里换一个sklearn库中现成的GradientBoostingClassifier，记为clf2

# In[17]:


from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier() 


# ## 训练

# In[18]:


clf2.fit(X_train, y_train) 


# ## 预测

# In[19]:


y_score = clf2.predict_proba(X_test)[:, clf2.classes_ == 1] 
y_pred = clf2.predict(X_test) 


# ## 评估

# In[20]:


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
logloss = log_loss(y_test, y_score)
print(acc, auc, logloss)


# **从测试集上的评估指标来看，模型clf2比clf1,clf1_p1好**

# # 模型迭代
# 将收集数据、特征工程、模型选择、模型参数选择、训练测试等步骤反复迭代，直到评价指标令人满意为止。

# 本文参考了 https://github.com/lenguyenthedat/kaggle-for-fun/blob/master/avazu-ctr-prediction/avazu-ctr-prediction.py
# 里面包括去除离群点、处理日期时间格式等操作
