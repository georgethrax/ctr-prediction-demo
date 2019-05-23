
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
```

    /home/lix/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
# 读取数据集
df = pd.read_csv("./ctr_data.csv", index_col=None)
df
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
    <tr>
      <th>5</th>
      <td>1.000072e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>d6137915</td>
      <td>bb1ef334</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>16920</td>
      <td>320</td>
      <td>50</td>
      <td>1899</td>
      <td>0</td>
      <td>431</td>
      <td>100077</td>
      <td>117</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000072e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>8fda644b</td>
      <td>25d4cfcd</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20362</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000092e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>e151e245</td>
      <td>7e091613</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20632</td>
      <td>320</td>
      <td>50</td>
      <td>2374</td>
      <td>3</td>
      <td>39</td>
      <td>-1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.000095e+19</td>
      <td>1</td>
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
      <td>15707</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.000126e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1002</td>
      <td>0</td>
      <td>84c7ba46</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>21689</td>
      <td>320</td>
      <td>50</td>
      <td>2496</td>
      <td>3</td>
      <td>167</td>
      <td>100191</td>
      <td>23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000187e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>e151e245</td>
      <td>7e091613</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17747</td>
      <td>320</td>
      <td>50</td>
      <td>1974</td>
      <td>2</td>
      <td>39</td>
      <td>100019</td>
      <td>33</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.000197e+19</td>
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
      <td>15701</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.000203e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>9e8cf15d</td>
      <td>0d3cb7be</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>100148</td>
      <td>157</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.000204e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>d6137915</td>
      <td>bb1ef334</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>19771</td>
      <td>320</td>
      <td>50</td>
      <td>2227</td>
      <td>0</td>
      <td>687</td>
      <td>100077</td>
      <td>48</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.000252e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>98fed791</td>
      <td>d9b5648e</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20984</td>
      <td>320</td>
      <td>50</td>
      <td>2371</td>
      <td>0</td>
      <td>551</td>
      <td>-1</td>
      <td>46</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.000354e+19</td>
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
      <td>15699</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.000359e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>d9750ee7</td>
      <td>98572c79</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17914</td>
      <td>320</td>
      <td>50</td>
      <td>2043</td>
      <td>2</td>
      <td>39</td>
      <td>-1</td>
      <td>32</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.000411e+19</td>
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
      <td>15708</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.000418e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>0c2fe9d6</td>
      <td>27e3c518</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>6558</td>
      <td>320</td>
      <td>50</td>
      <td>571</td>
      <td>2</td>
      <td>39</td>
      <td>-1</td>
      <td>32</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.000448e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>66a5f0f3</td>
      <td>d9b5648e</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>21234</td>
      <td>320</td>
      <td>50</td>
      <td>2434</td>
      <td>3</td>
      <td>163</td>
      <td>100088</td>
      <td>61</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.000451e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>543a539e</td>
      <td>c7ca3108</td>
      <td>3e814130</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20352</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.000457e+19</td>
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
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.000467e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>543a539e</td>
      <td>c7ca3108</td>
      <td>3e814130</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20366</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.000477e+19</td>
      <td>1</td>
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
      <td>15701</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.000525e+19</td>
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
      <td>100083</td>
      <td>79</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.000533e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1010</td>
      <td>1</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>ffc6ffd0</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>21665</td>
      <td>320</td>
      <td>50</td>
      <td>2493</td>
      <td>3</td>
      <td>35</td>
      <td>-1</td>
      <td>117</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.000554e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>e151e245</td>
      <td>7e091613</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20984</td>
      <td>320</td>
      <td>50</td>
      <td>2371</td>
      <td>0</td>
      <td>551</td>
      <td>100217</td>
      <td>46</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.000561e+19</td>
      <td>1</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>54c5d545</td>
      <td>2347f47a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>21611</td>
      <td>320</td>
      <td>50</td>
      <td>2480</td>
      <td>3</td>
      <td>297</td>
      <td>100111</td>
      <td>61</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.000565e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>543a539e</td>
      <td>c7ca3108</td>
      <td>3e814130</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20366</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.000595e+19</td>
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
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>1.138175e+19</td>
      <td>1</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>0a742914</td>
      <td>510bd839</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>19950</td>
      <td>320</td>
      <td>50</td>
      <td>1800</td>
      <td>3</td>
      <td>167</td>
      <td>100075</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>1.138178e+19</td>
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
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>1.138187e+19</td>
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
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>1.138190e+19</td>
      <td>1</td>
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
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>1.138195e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>f9c69707</td>
      <td>e16ceb4b</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>19665</td>
      <td>320</td>
      <td>50</td>
      <td>2253</td>
      <td>2</td>
      <td>303</td>
      <td>-1</td>
      <td>52</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>1.138217e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>bb4524e7</td>
      <td>d733bbc3</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20633</td>
      <td>320</td>
      <td>50</td>
      <td>2374</td>
      <td>3</td>
      <td>39</td>
      <td>100148</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>1.138244e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1010</td>
      <td>1</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>6103360b</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>21665</td>
      <td>320</td>
      <td>50</td>
      <td>2493</td>
      <td>3</td>
      <td>35</td>
      <td>-1</td>
      <td>117</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>1.138256e+18</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>f282ab5a</td>
      <td>61eb5bc4</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>1.138274e+18</td>
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
      <td>15702</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>1.138277e+19</td>
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
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>1.138288e+19</td>
      <td>1</td>
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
      <td>15701</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>1.138292e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>395fa97c</td>
      <td>3f797953</td>
      <td>3e814130</td>
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
    <tr>
      <th>9982</th>
      <td>1.138309e+19</td>
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
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>1.138310e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>9b124c1e</td>
      <td>db7a8013</td>
      <td>f028772b</td>
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
    <tr>
      <th>9984</th>
      <td>1.138318e+19</td>
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
      <td>15702</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>1.138328e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>030440fe</td>
      <td>08ba7db9</td>
      <td>76b2941d</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>1.138365e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>1779deee</td>
      <td>2347f47a</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>100162</td>
      <td>157</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>1.138376e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>d9750ee7</td>
      <td>98572c79</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17914</td>
      <td>320</td>
      <td>50</td>
      <td>2043</td>
      <td>2</td>
      <td>39</td>
      <td>100084</td>
      <td>32</td>
    </tr>
    <tr>
      <th>9988</th>
      <td>1.138381e+19</td>
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
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9989</th>
      <td>1.138384e+18</td>
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
      <td>15703</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>1.138396e+19</td>
      <td>1</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>d9750ee7</td>
      <td>98572c79</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17753</td>
      <td>320</td>
      <td>50</td>
      <td>1993</td>
      <td>2</td>
      <td>1063</td>
      <td>-1</td>
      <td>33</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>1.138399e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>f9c69707</td>
      <td>e16ceb4b</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>19665</td>
      <td>320</td>
      <td>50</td>
      <td>2253</td>
      <td>2</td>
      <td>303</td>
      <td>-1</td>
      <td>52</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>1.138400e+19</td>
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
      <td>15707</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100083</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>1.138405e+19</td>
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
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>1.138411e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>6256f5b4</td>
      <td>28f93029</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>16859</td>
      <td>320</td>
      <td>50</td>
      <td>1887</td>
      <td>3</td>
      <td>39</td>
      <td>-1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>1.138429e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>6c5b482c</td>
      <td>7687a86e</td>
      <td>3e814130</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17654</td>
      <td>300</td>
      <td>250</td>
      <td>1994</td>
      <td>2</td>
      <td>39</td>
      <td>100083</td>
      <td>33</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>1.138440e+19</td>
      <td>1</td>
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
      <td>15701</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>1.138442e+19</td>
      <td>1</td>
      <td>14102100</td>
      <td>1005</td>
      <td>1</td>
      <td>5ee41ff2</td>
      <td>17d996e6</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>16920</td>
      <td>320</td>
      <td>50</td>
      <td>1899</td>
      <td>0</td>
      <td>431</td>
      <td>-1</td>
      <td>117</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1.138466e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>5e3f096f</td>
      <td>2347f47a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>21611</td>
      <td>320</td>
      <td>50</td>
      <td>2480</td>
      <td>3</td>
      <td>297</td>
      <td>100111</td>
      <td>61</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>1.138513e+19</td>
      <td>0</td>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>83a0ad1a</td>
      <td>5c9ae867</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>18945</td>
      <td>320</td>
      <td>50</td>
      <td>2153</td>
      <td>3</td>
      <td>427</td>
      <td>100063</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 24 columns</p>
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


    /home/lix/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      



```python
X
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
    <tr>
      <th>5</th>
      <td>1005</td>
      <td>0</td>
      <td>236</td>
      <td>316</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16920</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1005</td>
      <td>0</td>
      <td>45</td>
      <td>218</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20362</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1005</td>
      <td>1</td>
      <td>153</td>
      <td>330</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20632</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>15707</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1002</td>
      <td>0</td>
      <td>244</td>
      <td>200</td>
      <td>5</td>
      <td>293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21689</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1005</td>
      <td>1</td>
      <td>153</td>
      <td>330</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17747</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15701</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1005</td>
      <td>0</td>
      <td>10</td>
      <td>242</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1005</td>
      <td>0</td>
      <td>236</td>
      <td>316</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19771</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1005</td>
      <td>0</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>187</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>20984</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15699</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1005</td>
      <td>0</td>
      <td>185</td>
      <td>322</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17914</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>15708</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1005</td>
      <td>1</td>
      <td>50</td>
      <td>15</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6558</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1005</td>
      <td>0</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>132</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>21234</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1005</td>
      <td>0</td>
      <td>248</td>
      <td>127</td>
      <td>4</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20352</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>21</th>
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
      <th>22</th>
      <td>1005</td>
      <td>0</td>
      <td>248</td>
      <td>127</td>
      <td>4</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20366</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15701</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>24</th>
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
      <th>25</th>
      <td>1010</td>
      <td>1</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>312</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>21665</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1005</td>
      <td>1</td>
      <td>153</td>
      <td>330</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20984</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1005</td>
      <td>0</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>105</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>21611</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1005</td>
      <td>0</td>
      <td>248</td>
      <td>127</td>
      <td>4</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20366</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>29</th>
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
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>1005</td>
      <td>1</td>
      <td>91</td>
      <td>13</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19950</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>1005</td>
      <td>0</td>
      <td>283</td>
      <td>367</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19665</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>1005</td>
      <td>0</td>
      <td>267</td>
      <td>282</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20633</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>1010</td>
      <td>1</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>122</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>21665</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>1005</td>
      <td>0</td>
      <td>112</td>
      <td>357</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15702</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15701</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>1005</td>
      <td>0</td>
      <td>71</td>
      <td>80</td>
      <td>4</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18993</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9982</th>
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
      <th>9983</th>
      <td>1005</td>
      <td>0</td>
      <td>272</td>
      <td>240</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18993</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9984</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15702</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>1005</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>1005</td>
      <td>0</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>23</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>20596</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>1005</td>
      <td>0</td>
      <td>185</td>
      <td>322</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17914</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9988</th>
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
      <th>9989</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15703</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>1005</td>
      <td>1</td>
      <td>185</td>
      <td>322</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17753</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>1005</td>
      <td>0</td>
      <td>283</td>
      <td>367</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19665</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15707</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>1005</td>
      <td>0</td>
      <td>52</td>
      <td>147</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16859</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>1005</td>
      <td>0</td>
      <td>143</td>
      <td>159</td>
      <td>4</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17654</td>
      <td>300</td>
      <td>250</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>1005</td>
      <td>0</td>
      <td>301</td>
      <td>43</td>
      <td>2</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15701</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>1005</td>
      <td>1</td>
      <td>27</td>
      <td>141</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16920</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1005</td>
      <td>0</td>
      <td>244</td>
      <td>202</td>
      <td>5</td>
      <td>119</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>21611</td>
      <td>320</td>
      <td>50</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>1005</td>
      <td>0</td>
      <td>105</td>
      <td>197</td>
      <td>12</td>
      <td>293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18945</td>
      <td>320</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 12 columns</p>
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

    0.819 0.6646154575328616 2.032632528109397


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

    /home/lix/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





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

    0.8225 0.687026914110774 0.4252496557629438


**从测试集上的评估指标来看，模型clf2比clf1,clf1_p1好**

# 模型迭代
将收集数据、特征工程、模型选择、模型参数选择、训练测试等步骤反复迭代，直到评价指标令人满意为止。

本文参考了 https://github.com/lenguyenthedat/kaggle-for-fun/blob/master/avazu-ctr-prediction/avazu-ctr-prediction.py
里面包括去除离群点、处理日期时间格式等操作
