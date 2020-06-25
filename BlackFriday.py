#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# In[7]:


train= pd.read_csv('C:/Users/SAM/OneDrive/Desktop/Project/Kaggle Proj/Black Friday/train.csv')
test= pd.read_csv('C:/Users/SAM/OneDrive/Desktop/Project/Kaggle Proj/Black Friday/test.csv')


# In[8]:


train.head()


# In[9]:


train.isnull().sum()


# In[11]:


# Combine train and test to do encoding of categorical variables
frames = [train, test]
input = pd.concat(frames)

print(input.shape)
input.head()


# In[12]:


input.dtypes


# In[13]:


#Replace missing values with -999

input.fillna(999, inplace=True)


# In[15]:


input.head()


# In[16]:


#Create target column
target = input.Purchase


# In[17]:


target = np.array(target)


# In[18]:


#Drop purchase from input
input.drop(["Purchase"], axis=1, inplace=True)


# In[20]:


print(input.columns, input.dtypes)


# In[21]:


#Convert all the columns to string 
input = input.applymap(str)
input.dtypes


# In[22]:


# Have a copy of the pandas dataframe. Will be useful later on
inputcopy = input.copy()


# In[23]:


#Convert categorical to numeric using LabelEncoder

input = np.array(input)

for i in range(input.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(input[:,i]))
    input[:, i] = lbl.transform(input[:, i])


# In[24]:


input.dtypes


# In[27]:


input = input.astype(int)


# In[28]:


# Split dataset into two. First level models to create meta features to feed into a second level model
first_stage_rows = np.random.randint(train.shape[0], size = np.int(train.shape[0]/2))


# In[29]:



train_np   = input[:train.shape[0], :]
target_np  = target[:train.shape[0]]
train_fs   = train_np[first_stage_rows, :]
target_fs  = target_np[first_stage_rows]
train_ss   = train_np[-first_stage_rows, :]
target_ss  = target_np[-first_stage_rows]


# In[31]:


print(train_fs.shape, target_fs.shape, train_ss.shape, target_ss.shape)


# In[32]:


train_fs


# In[33]:



train_ss


# In[34]:


xgtrain = xgb.DMatrix(train_fs, label=target_fs)
watchlist = [(xgtrain, 'train')]

# Model 1: 6/3000

params = {}
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 6
params["nthread"] = 6
#params["gamma"] = 1
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["base_score"] = 1800
params["eval_metric"] = "rmse"
params["seed"] = 0

plst = list(params.items())
num_rounds = 3000

model_1 = xgb.train(plst, xgtrain, num_rounds)

# Model 2: 8/1420

params = {}
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["nthread"] = 6
#params["gamma"] = 1
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["base_score"] = 1800
params["eval_metric"] = "rmse"
params["seed"] = 0

plst = list(params.items())
num_rounds = 1420

model_2 = xgb.train(plst, xgtrain, num_rounds)

# Model 3: 10/1200

params = {}
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 10
params["nthread"] = 6
#params["gamma"] = 1
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["base_score"] = 1800
params["eval_metric"] = "rmse"
params["seed"] = 0

plst = list(params.items())
num_rounds = 1200

model_3 = xgb.train(plst, xgtrain, num_rounds)

# Model 4: 12/800

params = {}
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 12
params["nthread"] = 6
#params["gamma"] = 1
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["base_score"] = 1800
params["eval_metric"] = "rmse"
params["seed"] = 0

plst = list(params.items())
num_rounds = 800

model_4 = xgb.train(plst, xgtrain, num_rounds)


# In[35]:


# This set of models will be ExtraTrees

# Model 5: 8/1450

model_5 = ExtraTreesRegressor(n_estimators=1450, 
                              max_depth=8,
                              min_samples_split=10, 
                              min_samples_leaf=10, 
                              oob_score=True, 
                              n_jobs=6, 
                              random_state=123, 
                              verbose=1, 
                              bootstrap=True)
model_5.fit(train_fs, target_fs)

# Model 6: 6/3000

model_6 = ExtraTreesRegressor(n_estimators=3000, 
                              max_depth=6,
                              min_samples_split=10, 
                              min_samples_leaf=10, 
                              oob_score=True, 
                              n_jobs=6, 
                              random_state=123, 
                              verbose=1, 
                              bootstrap=True)
model_6.fit(train_fs, target_fs)

# Model 7: 12/800

model_7 = ExtraTreesRegressor(n_estimators=800, 
                              max_depth=12,
                              min_samples_split=10, 
                              min_samples_leaf=10, 
                              oob_score=True, 
                              n_jobs=6, 
                              random_state=123, 
                              verbose=1, 
                              bootstrap=True)
model_7.fit(train_fs, target_fs)


# In[36]:


# This set of models will be RandomForest

# Model 8: 6/3000
model_8 = RandomForestRegressor(n_estimators=3000, max_depth=6, oob_score=True, n_jobs=6, random_state=123, min_samples_split=10, min_samples_leaf=10)
model_8.fit(train_fs, target_fs)

# Model 9: 8/1500
model_9 = RandomForestRegressor(n_estimators=1500, max_depth=8, oob_score=True, n_jobs=6, random_state=123, min_samples_split=10, min_samples_leaf=10)
model_9.fit(train_fs, target_fs)

# Model 10: 12/800
model_10 = RandomForestRegressor(n_estimators=800, max_depth=12, oob_score=True, n_jobs=6, random_state=123, min_samples_split=10, min_samples_leaf=10)
model_10.fit(train_fs, target_fs)


# In[37]:


model_1_predict = model_1.predict(xgb.DMatrix(train_ss))
model_2_predict = model_2.predict(xgb.DMatrix(train_ss))
model_3_predict = model_3.predict(xgb.DMatrix(train_ss))
model_4_predict = model_4.predict(xgb.DMatrix(train_ss))
model_5_predict = model_5.predict(train_ss)
model_6_predict = model_6.predict(train_ss)
model_7_predict = model_7.predict(train_ss)
model_8_predict = model_8.predict(train_ss)
model_9_predict = model_9.predict(train_ss)
model_10_predict = model_10.predict(train_ss)


# In[38]:



train_ss_w_meta = np.concatenate((train_ss, np.vstack((model_1_predict, model_2_predict, model_3_predict, 
                                                       model_4_predict, model_5_predict,
              model_6_predict, model_7_predict, model_8_predict, model_9_predict, model_10_predict)).T), axis=1)


# In[41]:



kfolds = KFold(train_ss_w_meta.shape[0],n_splits=5)


# In[42]:



params = {}
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["nthread"] = 6
#params["gamma"] = 1
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["base_score"] = 1800
params["eval_metric"] = "rmse"
params["seed"] = 0

plst = list(params.items())
num_rounds = 1400


# In[45]:


for train_index, validation_index in kfolds:
    
    train_X, validation_X = train_ss_w_meta[train_index, :], train_ss_w_meta[validation_index, :]
    train_y, validation_y = target_ss[train_index], target_ss[validation_index]
    
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain, 'train')]
    model_cv_xgboost = xgb.train(plst, xgtrain, num_rounds)
    model_cv_predict = model_cv_xgboost.predict(xgb.DMatrix(validation_X))
    print(np.sqrt(mean_squared_error(validation_y, model_cv_predict)))


# In[47]:



# RMSE is around 2050.
# Training second stage model on all the second stage data now

xgtrain = xgb.DMatrix(train_ss_w_meta, label=target_ss)
watchlist = [(xgtrain, 'train')]
model_ss_xgboost = xgb.train(plst, xgtrain, num_rounds)


# In[48]:


model_1_predict = model_1.predict(xgb.DMatrix(input[train.shape[0]:, :]))
model_2_predict = model_2.predict(xgb.DMatrix(input[train.shape[0]:, :]))
model_3_predict = model_3.predict(xgb.DMatrix(input[train.shape[0]:, :]))
model_4_predict = model_4.predict(xgb.DMatrix(input[train.shape[0]:, :]))
model_5_predict = model_5.predict(input[train.shape[0]:, :])
model_6_predict = model_6.predict(input[train.shape[0]:, :])
model_7_predict = model_7.predict(input[train.shape[0]:, :])
model_8_predict = model_8.predict(input[train.shape[0]:, :])
model_9_predict = model_9.predict(input[train.shape[0]:, :])
model_10_predict = model_10.predict(input[train.shape[0]:, :])

test_ss_w_meta = np.concatenate((input[train.shape[0]:, :], np.vstack((model_1_predict, model_2_predict, model_3_predict, 
                                                       model_4_predict, model_5_predict,
              model_6_predict, model_7_predict, model_8_predict, model_9_predict, model_10_predict)).T), axis=1)


# In[49]:


model_ss_predict = model_ss_xgboost.predict(xgb.DMatrix(test_ss_w_meta))


# In[50]:


np.max(model_ss_predict), np.min(model_ss_predict)


# In[53]:


submission.Purchase = model_ss_predict


# In[52]:


submission = pd.read_csv("C:/Users/SAM/OneDrive/Desktop/Project/Kaggle Proj/Black Friday/submission.csv")


# In[54]:


submission.to_csv("submit.csv", index=False)


# In[ ]:




