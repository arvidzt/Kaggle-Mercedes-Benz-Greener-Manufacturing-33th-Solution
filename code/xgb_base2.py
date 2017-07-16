
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn import cluster


# ### Import

# In[2]:

# read datasets
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')

test.drop(['y'],axis = 1,inplace=True)


# In[62]:

for c in train.columns:
    if c not in train.columns[:11]:
        if  ((train[c].sum() < 1)):
            train.drop(c,axis = 1,inplace = True)
            test.drop(c,axis = 1,inplace = True)

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))



col = [k for k in train.columns if k not in {"y","X0","X1","X2","X3","X4","X5","X6","X8"}]


n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y","X0","X5"], axis=1))
pca2_results_test = pca.transform(test.drop(["X0","X5"], axis=1))

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    #train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    #test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]


# In[71]:

n_comp = 10
# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42)
grp_results_train = grp.fit_transform(train.drop(["y","X0","X5"], axis=1))
grp_results_test = grp.transform(test.drop(["X0","X5"], axis=1))

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=42)
srp_results_train = srp.fit_transform(train.drop(["y","X0","X5"], axis=1))
srp_results_test = srp.transform(test.drop(["X0","X5"], axis=1))

for i in range(1, n_comp+1):
    train['grp_' + str(i)] = grp_results_train[:,i-1]
    test['grp_' + str(i)] = grp_results_test[:, i-1]

    train['srp_' + str(i)] = srp_results_train[:,i-1]
    test['srp_' + str(i)] = srp_results_test[:, i-1]


# In[72]:

n_comp = 10
# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y","X0","X5"], axis=1))
ica2_results_test = ica.transform(test.drop(["X0","X5"], axis=1))
for i in range(1, n_comp+1):
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]




def GRR(train,test,num):
    X2_train = train.groupby('X'+str(num))['y'].mean().reset_index()
    X2_train.rename(columns={"y":'X'+str(num)+'G'},inplace= True)

    X2_test = train.groupby('X'+str(num))['y'].mean().reset_index()
    X2_test.rename(columns={"y":'X'+str(num)+'G'},inplace= True)

    train = pd.merge(train,X2_train,how ='left',on = 'X'+str(num))
    test = pd.merge(test,X2_test,how = 'left',on = 'X'+str(num))

    test['X'+str(num)+'G'] = test['X'+str(num)+'G'].fillna(test['X'+str(num)+'G'].mean())
    return train,test


# In[75]:

def GRM(train,test,num):
    X2_train = train.groupby('X'+str(num))['y'].median().reset_index()
    X2_train.rename(columns={"y":'X'+str(num)+'M'},inplace= True)

    X2_test = train.groupby('X'+str(num))['y'].median().reset_index()
    X2_test.rename(columns={"y":'X'+str(num)+'M'},inplace= True)

    train = pd.merge(train,X2_train,how ='left',on = 'X'+str(num))
    test = pd.merge(test,X2_test,how = 'left',on = 'X'+str(num))

    test['X'+str(num)+'M'] = test['X'+str(num)+'M'].fillna(test['X'+str(num)+'M'].mean())
    return train,test



train,test = GRR(train,test,0)


test[test.X0G.isnull()]


# In[80]:

test.to_csv('last_test.csv',index=False)
train.to_csv('last_train.csv',index=False)


import xgboost as xgb

y_train = train["y"]
y_mean = np.mean(y_train)

# prepare dict of params for xgboost to run with
xgb_params = {
   'n_trees': 520,
   'eta': 0.0045,
   'max_depth': 4,
   'subsample': 0.93,
   'objective': 'reg:linear',
   #'eval_metric': 'rmse',
   'base_score': y_mean, # base prediction = mean(target)
   'silent': 1
}

def xgb_r2_score(preds, dtrain):
   labels = dtrain.get_label()
   return 'r2', r2_score(labels, preds)

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                  dtrain,
                  num_boost_round=1000, # increase to have better results (~700)
                  feval=xgb_r2_score,
                  early_stopping_rounds=50,
                  verbose_eval=50,
                  show_stdv=False
                 )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
fig, ax = plt.subplots(1, 1, figsize=(13, 20))
xgb.plot_importance(model, height=0.5, ax=ax)


# In[84]:

# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))


# In[87]:

y_pred = model.predict(dtest)

# make predictions and save results

output = pd.DataFrame({'ID': test['ID'].astype(np.int32), 'y2': y_pred})


# In[88]:

output.to_csv('xgboost_base2.csv', index=False)


# In[90]:

res1 = pd.read_csv("../xgboost2.csv")
res2 = pd.read_csv("./xgboost_base2.csv")

res = res1.merge(res2,on = 'ID',suffixes=['_1','_2'])

res['y'] = 0.5 * res.y_1 +0.5 * res.y_2

res.drop(['y_1','y_2'],axis = 1,inplace=True)

res.to_csv("resemble2.csv",index = False)



res1 = pd.read_csv("./resemble1.csv")
#res3 = pd.read_csv("./stacked-models.csv")
res2 = pd.read_csv("./resemble2.csv")
res3 = pd.read_csv("./subXgb_Stack_Stack_No_ID.csv")


res = res1.merge(res2,on = 'ID',suffixes=['_1','_2']).merge(res3,on = 'ID')

res['y'] = 0.5 * res.y_1 +0.4 * res.y_2 + 0.1*res.y

res.drop(['y_1','y_2'],axis = 1,inplace=True)

res.to_csv("resemble3.csv",index = False)
