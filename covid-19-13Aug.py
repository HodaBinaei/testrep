#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
# OS
import os
# Pandas
import pandas as pd
# Numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
from os.path import join
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt   
from sklearn import preprocessing
##Metrics
from sklearn.metrics import mean_squared_error
# 1.3 Regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from collections import OrderedDict
from operator import itemgetter 
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


# In[30]:







def _mlmodels():
    dirpath = input("please enter the dir of the data then insert /name ")
    df = pd.read_csv(dirpath)
    df = pd.DataFrame(df)
    X=df.iloc[:, 0:4]
    y=df.iloc[:, 4:11]
    X=X.rename_axis('ID').values
    y=y.rename_axis('ID').values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
    ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor()}

    results = {}
    names = []
    for name, model in ESTIMATORS.items():
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)
        print( " RMSE of %s model for all: %f " % (name,
                                               round (sqrt(mean_squared_error(y_test, ypred)),3)))
        results.update({ name :round (sqrt(mean_squared_error(y_test, ypred)),3) }) 
        
        data =   pd.DataFrame({'q5': np.round(ypred[: ,0],3) ,
                           'q10':np.round(ypred[: ,1],3),
                            'q25':np.round(ypred[: ,2],3),   
                           'q50': np.round(ypred[: ,3],3),
                            'q75':np.round(ypred[: ,4],3),   
                           'q90': np.round(ypred[: ,5],3), 
                             'q95':np.round(ypred[: ,6] ,3)})    
    print( data)
    


# In[31]:


_mlmodels()




#/home/hoda/Desktop/out_q/out_lt1.txt


# In[ ]:





# In[ ]:





# In[32]:


_mlmodels()




#/home/hoda/Desktop/out_q/out_lt1.txt


# In[28]:


_mlmodels()


# In[ ]:





# In[55]:


_mlmodels()


# In[ ]:


home(/hoda/Desktop/out_q/out_lt1.txt)


# In[38]:


_mlmodels()


# In[2]:



# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
df = pd.read_csv (r'/home/hoda/Desktop/rstudio-export/out_gt1.txt')
df.to_csv(r'/home/hoda/Desktop/rstudio-export/out_eq1.gtt.csv')
 
X=df.iloc[:, 0:4]
y=df.iloc[:, 4:11]
X=X.rename_axis('ID').values
y=y.rename_axis('ID').values





X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)



ESTIMATORS = {"Extra trees": ExtraTreesRegressor( ) }

results = {}
names = []
scoring = 'accuracy'
for name, model in ESTIMATORS.items():
    model.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk


loaded_model = joblib.load(filename)
ypred = model.predict(X_test)
print( " RMSE of %s model for gt1: %f " % (name,
                                               round (sqrt(mean_squared_error(y_test, ypred)),3)))
results.update({ name :round (sqrt(mean_squared_error(y_test, ypred)),3) })  
    

data =   pd.DataFrame({'q5': np.round(ypred[: ,0],3) ,
                           'q10':np.round(ypred[: ,1],3),
                            'q25':np.round(ypred[: ,2],3),   
                           'q50': np.round(ypred[: ,3],3),
                            'q75':np.round(ypred[: ,4],3),   
                           'q90': np.round(ypred[: ,5],3), 
                             'q95':np.round(ypred[: ,6] ,3)})    
print( data)


# In[ ]:




