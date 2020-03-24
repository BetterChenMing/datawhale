# Task2:数据的探索性分析（EDA）

1、分析提供的变量与需预测值间之间可能存在的关系，以及是否可以自建特征，可绘制图表。
2、赛题地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281949.1003.2.493e2448KgHsEd

一、赛题数据
赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。

## EDA的目标
1、熟悉已知数据集，对数据集进行查看与验证来确定所获得的数据集可以用于接下来的机器学习使用；
2、了解了数据集之后，下一步要去了解变量间的相互关系以及变量与预测值之间存在的关系；
3、引导数据科学从业者进行数据处理以及特征工程的步骤，让接下来的预测问题更可靠；

### 具体步骤：

1、导入需要用到的python库与包

```
# jupyter magic command,there are so many magic like this.
%config ZMQInteractiveShell.ast_node_interactivity='all' # 打印单元格内所有输出
```
```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # 缺失值的可视化处理
```
2、分别载入训练集和测试集数据，载入数据后，可以通过head(),shape(),tail()等方式来初步观察数据
```
"""导入数据集"""
train_data = pd.read_csv('data/used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv('data/used_car_testA_20200313.csv', sep=' ')
```
```
train_data.shape # (150000, 31)
test_data.shape # (50000, 30)
```
可以看出训练集共 150000 个样本，30 个特征，1 列价格；测试集共 50000 个样本，30 个特征
```
train_data.head()
train_data.info()
train_data.describe()
```
通过info()可以发现字段的type，其中 notRepairedDamage 为 object类型,其次部份字段有缺失值，可能需要填充处理（方法待确定）
```
train_data['notRepairedDamage'].value_counts()  #数据有非数字的值 ‘-’存在，考虑置为 na 或者用中位数替代（待实现）
```

```
test_data.head()
test_data.info()
test_data.describe()
```
3、了解我们要预测的对象的分布情况
```
"""查看预测值的频数"""
train_data['price'].value_counts()

# 直方图可视化 自动划分10（默认值）个价格区间 统计每个区间的频数
plt.hist(train_data['price'], orientation='vertical', histtype='bar', color='red')
plt.show()
```
发现价格大于 20000 的值极少，可以考虑把这些值当作离群点（或异常值）直接剔除（不确定，回归问题考虑可以保留）
```
import scipy.stats as st
y = train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```
由拟合结果可以看出，价格并不服从正态分布（单独汽车这个类别不奇怪），在进行回归分析之前需要转换。虽然对数变换做得很好，但最佳拟合还是无界约翰逊分布。

