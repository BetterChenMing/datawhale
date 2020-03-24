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
3、了解我们要预测的对象的分布情况,将字段分成数值型和类别型，后面分开查看和处理
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
```
"""查看偏度和峰度"""
sns.distplot(Train_data['price'])
print("Skewness: %f" % Train_data['price'].skew())
print("Kurtosis: %f" % Train_data['price'].kurt())
# train_data.skew()
# train_data.kurt()
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.distplot(train_data.skew(), color='blue', axlabel='Skewness')
plot.subplot(122)
sns.distplot(train_data.kurt(), color='orange', axlabel='Kurtness')
```
把字段分为数值字段和类别字段
```
"""先分离出label值"""
y_train = train_data['price']

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

```
4、接下来我们主要对数值特征和类别特征进一步挖掘信息，包括类别偏斜，类别分布可视化，数值可视化等
```
"""类别偏斜处理"""
for cat_fea in categorical_features:
    print(cate_fea + '特征分布如下：')
    print('{}特征有{}不同的值'.format(cate_fea, len(train_data[cat_fea].unique())))
    print(train_data[cat_fea].value_counts())
    print()
# seller、offerType等字段偏斜就比较严重，直接删除这些字段
del train_data['seller']
del train_data['offerType']
del test_data['seller']
del test_data['offerType']

categorical_features.remove('seller')
categorical_features.remove('offerType')
```
```
"""类别的unique分布"""
for cat in categorical_features:
    print(len(train_data[cat].unique()))

# 结果
249
40
9
8
3
3
7905

# 因为regionCode的类别太稀疏了，所以先去掉，因为后面要可视化，不画稀疏的
categorical_features.remove('regionCode')
```
```
"""类别特征箱型图可视化"""
for c in categorical_features:
    train_data[c] = train_data[c].astype('category')
    if train_data[c].isnull().any():
        train_data[c] = train_data[c].cat.add_categories(['MISSING'])
        train_data[c] = train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxenplot(x=x, y=y)
    x = plt.xticks(rotation=90)

f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
```
```
"""类别特征的小提琴图可视化， 小提琴图类似箱型图，比后者高级点，图好看些"""
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=train_data)
    plt.show()

"""类别特征的柱形图可视化"""
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")
```
```
"""类别特征的每个类别频数可视化(count_plot)"""
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
```
数值特征的探索我们要分析相关性等
```
numeric_train_data = train_data[numeric_features]

# 把price这一列加上，这个也是数值
numeric_train_data['price'] = Y_train

"""相关性分析"""
correlation = numeric_train_data.corr()
print(correlation['price'].sort_values(ascending=False), '\n')   # 与price相关的特征排序
```
```
# 可视化
f, ax = plt.subplots(figsize=(10,10))
plt.title('Correlation of Numeric Features with Price', y=1, size=16)
sns.heatmap(correlation, square=True, vmax=0.8)
```
```
# 删除price
del numeric_train_data['price']

"""查看几个数值特征的偏度和峰度"""
for col in numeric_train_data.columns:
     print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(numeric_train_data[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(numeric_train_data[col].kurt())  
         )

"""每个数字特征得分布可视化"""
f = pd.melt(train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```

```
"""数字特征相互之间的关系可视化"""
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
# 这里可以看到有些特征之间是相关的， 比如 v_1 和 v_6
```













