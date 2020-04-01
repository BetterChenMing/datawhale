# Task4 modeling_and_tunning
```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

import seaborn as sns

import datetime
```


```python
## 这个方法很有用呀，学习到了，可以合理有效的优化数据的内存使用，赞！
```


```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) /
                                        start_mem))
    return df
```


```python
# 读取数据并使用优化内存占用方法，效果显著
sample_feature = reduce_mem_usage(pd.read_csv('data/data_for_tree.csv'))
```


```python
continuous_feature_names = [
    x for x in sample_feature.columns if x not in ['price', 'brand', 'model']
]
```


```python
# 线性回归 & 五折交叉验证 & 模拟真实业务情况
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature[
    'notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']
```


```python
model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
```


```python
'intercept:' + str(model.intercept_)

sorted(dict(zip(continuous_feature_names, model.coef_)).items(),
       key=lambda x: x[1],
       reverse=True)
```


```python
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
```


```python
plt.scatter(train_X['v_9'][subsample_index],
            train_y[subsample_index],
            color='black')
plt.scatter(train_X['v_9'][subsample_index],
            model.predict(train_X.loc[subsample_index]),
            color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price', 'Predicted Price'], loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()

# 绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布差异较大，且部分预测值出现了小于0的情况，说明我们的模型存在一些问题
```


```python
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(train_y)
plt.subplot(1, 2, 2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
```


```python
train_y_ln = np.log(train_y + 1)

print('The transformed price seems like normal distribution')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(train_y_ln)
plt.subplot(1, 2, 2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
```


```python
model = model.fit(train_X, train_y_ln)

print('intercept:' + str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(),
       key=lambda x: x[1],
       reverse=True)
```


```python
# 再次进行可视化，发现预测结果与真实值较为接近，且未出现异常状况
plt.scatter(train_X['v_9'][subsample_index],
            train_y[subsample_index],
            color='black')
plt.scatter(train_X['v_9'][subsample_index],
            np.exp(model.predict(train_X.loc[subsample_index])),
            color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price', 'Predicted Price'], loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()
```


```python

```


```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
```


```python
def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result

    return wrapper
```


```python
scores = cross_val_score(model,
                         X=train_X,
                         y=train_y,
                         verbose=1,
                         cv=5,
                         scoring=make_scorer(
                             log_transfer(mean_absolute_error)))
```


```python
print('AVG:', np.mean(scores))
```


```python
scores = cross_val_score(model,
                         X=train_X,
                         y=train_y_ln,
                         verbose=1,
                         cv=5,
                         scoring=make_scorer(mean_absolute_error))
```


```python
print('AVG:', np.mean(scores))
```


```python
scores = pd.DataFrame(scores.reshape(1, -1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
scores
```


```python

```


```python
sample_feature = sample_feature.reset_index(drop=True)
```


```python
split_point = len(sample_feature) // 5 * 4
```


```python
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)
```


```python
model = model.fit(train_X, train_y_ln)
mean_absolute_error(val_y_ln, model.predict(val_X))
```


```python
from sklearn.model_selection import learning_curve, validation_curve
```


```python
def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_size=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_size,
        scoring=make_scorer(mean_absolute_error))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()  #区域
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             'o-',
             color='r',
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             'o-',
             color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt
```


```python
plot_learning_curve(LinearRegression(),
                    'Liner_model',
                    train_X[:1000],
                    train_y_ln[:1000],
                    ylim=(0.0, 0.5),
                    cv=5,
                    n_jobs=1)
```


```python
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)
```


```python

```


```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```


```python
models = [LinearRegression(), Ridge(), Lasso()]
```


```python
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model,
                             X=train_X,
                             y=train_y_ln,
                             verbose=0,
                             cv=5,
                             scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
```


```python
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
```


```python
model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```


```python
model = Ridge().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```


```python
model = Lasso().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```


```python

```


```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
```


```python
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor(solver='lbfgs', max_iter=100),
    XGBRegressor(n_estimators=100, objective='reg:squarederror'),
    LGBMRegressor(n_estimators=100)
]
```


```python
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model,
                             X=train_X,
                             y=train_y_ln,
                             verbose=0,
                             cv=5,
                             scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
```


```python
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
```


```python
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3, 5, 10, 15, 20, 40, 55]
max_depth = [3, 5, 10, 15, 20, 40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []
```


```python
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(
        cross_val_score(model,
                        X=train_X,
                        y=train_y_ln,
                        verbose=0,
                        cv=5,
                        scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score

best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(),
                                        key=lambda x: x[1])[0],
                          num_leaves=leaves)
    score = np.mean(
        cross_val_score(model,
                        X=train_X,
                        y=train_y_ln,
                        verbose=0,
                        cv=5,
                        scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score

best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(),
                                        key=lambda x: x[1])[0],
                          num_leaves=min(best_leaves.items(),
                                         key=lambda x: x[1])[0],
                          max_depth=depth)
    score = np.mean(
        cross_val_score(model,
                        X=train_X,
                        y=train_y_ln,
                        verbose=0,
                        cv=5,
                        scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score
```


```python
sns.lineplot(
    x=['0_initial', '1_turning_obj', '2_turning_leaves', '3_turning_depth'],
    y=[
        0.143,
        min(best_obj.values()),
        min(best_leaves.values()),
        min(best_depth.values())
    ])
```


```python
# 网格搜索
from sklearn.model_selection import GridSearchCV
```


```python
parameters = {
    'objective': objective,
    'num_leaves': num_leaves,
    'max_depth': max_depth
}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
```


```python
clf.best_params_
```


```python
model = LGBMRegressor(objective='regression', num_leaves=55, max_depth=15)
```


```python
np.mean(
    cross_val_score(model,
                    X=train_X,
                    y=train_y_ln,
                    verbose=0,
                    cv=5,
                    scoring=make_scorer(mean_absolute_error)))
```


```python
from bayes_opt import BayesianOptimization
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-7-93303b4629f3> in <module>
    ----> 1 from bayes_opt import BayesianOptimization
    

    ModuleNotFoundError: No module named 'bayes_opt'



```python
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(LGBMRegressor(
        objective='regression_l1',
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        subsample=subsample,
        min_child_samples=int(min_child_samples)),
                          X=train_X,
                          y=train_y_ln,
                          verbose=0,
                          cv=5,
                          scoring=make_scorer(mean_absolute_error)).mean()
    return 1 - val
```


```python
rf_bo = BayesianOptimization(
    rf_cv, {
        'num_leaves': (2, 100),
        'max_depth': (2, 100),
        'subsample': (0.1, 1),
        'min_child_samples': (2, 100)
    })
```


```python
rf_bo.maximize()
```


```python
1 - rf_bo.max['target']
```


```python
plt.figure(figsize=(13, 5))
sns.lineplot(x=[
    '0_origin', '1_log_transfer', '2_L1_&_L2', '3_change_model',
    '4_parameter_turning'
],
             y=[1.36, 0.19, 0.19, 0.14, 0.13])
```
