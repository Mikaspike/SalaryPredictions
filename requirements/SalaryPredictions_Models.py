#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author
__author__ = "Michael Tin"
__email__ = "michael.tslin@gmail.com"


# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


# In[277]:


#functions
def load_file(file):
    '''loads .csv file to pandas dataframe'''
    return pd.read_csv(file)

#merging two dataframes by joining on common key      
def merge_data(df1, df2, key=None, left_index=False, right_index=False):
    '''perform inner join to return only records that are present in both dataframes'''
    return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)

#checking for duplicates
def check_duplicates(df):
    '''returns the duplicates from the dataframe'''
    return df[df.duplicated()]

def Normalize_df(df,num_list=None):
    '''normalizes numerical features with minmax scaling and combines results with categorical features
    cat_list and num_list must be predefined'''
    num_df = df[num_list]
    num_df = MinMaxScaler().fit_transform(num_df)
    df[num_list] = num_df
    return df

def one_hot_encode_feature_df(df, cat_list=None, num_list=None):
    '''performs one-hot encoding on all categorical variables and combines result with continous variables
    cat_list and num_list to be created first'''
    cat_df = pd.get_dummies(df[cat_list],dtype=int)
    num_df = df[num_list].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)#,ignore_index=False)

def label_encode_feature_df(df,cat_list=None,num_list=None):
    '''performs label encoding on all categorical variables and combines results with the continous variables
    cat_list and num_list to be created first'''
    cat_df = df[cat_list]
    for col in cat_df:
        cat_df[col] =  LabelEncoder().fit_transform(cat_df[col])
    num_df = df[num_list].apply(pd.to_numeric)
    return pd.concat([cat_df,num_df],axis=1)


def train_model(model, feature_df, target_df, num_procs, eval_metric, cv_std):
    '''Trains model with on Mean Squared Error scoring'''
    cv_results = cross_val_score(model, feature_df, target_df, cv=folds, n_jobs=num_procs, scoring=scoring)
    eval_metric[model] = -1.0*np.mean(cv_results)
    cv_std[model] = np.std(cv_results)

def print_summary(model, mean_mse, cv_std):
    print('\nModel:\n', model)
    print('Average MSE:\n', eval_metric[model])
    print('Standard deviation during CV:\n', cv_std[model])

    


# In[11]:


#Loading data
df_train = load_file('data/train_features.csv')
df_salaries = load_file('data/train_salaries.csv')
df_test = load_file('data/test_features.csv')

#define variables
cat_list = ['companyId', 'jobType', 'degree', 'major', 'industry']
num_list = ['yearsExperience', 'milesFromMetropolis']
target_col = 'salary'

#consolidating training dataset
df_train_raw = merge_data(df_train, df_salaries, key='jobId', left_index=False, right_index=False)

#removing duplicates - drop 'jobId first' as we have done in EDA
df_train_raw =df_train_raw.drop(columns=['jobId'])
duplic = check_duplicates(df_train_raw)
df_train_raw.drop(duplic.index, inplace=True)

#removing outliers
df_train_raw = df_train_raw[df_train_raw.salary > 8.5]

#shuffle dataframe and reset index to improve cross-validation accuracy
df_train_raw = shuffle(df_train_raw).reset_index()


# In[ ]:


#Normalising numerical variables on training and test datasets with MinMax scaler
df_train_raw = Normalize_df(df_train_raw,num_list=num_list)
df_test = Normalize_df(df_test,num_list=num_list)


# In[200]:


df_train_raw.head()


# In[194]:


df_test.head()


# In[289]:


#encoding categorical variables and get final dfs
df_train_final = one_hot_encode_feature_df(df_train_raw,cat_list=cat_list,num_list=num_list)
df_test_final = one_hot_encode_feature_df(df_test,cat_list=cat_list,num_list=num_list)


# In[354]:


#initialize model list and dicts
models=[]
eval_metric={}
cv_std={}
folds = 5

#define number of processes to run in parallel
num_procs = 2

#shared model paramaters
verbose_lvl = 0


# ## Evaluation of models - 2 linear and 2 non-linear

# In[355]:


#Generate models and print results - hyperparameter tuning done manually for each model

lr = LinearRegression()
rd = Ridge()
rf = RandomForestRegressor(n_estimators=150, n_jobs=num_procs, max_depth=25, min_samples_split=60,                            max_features=30, verbose=verbose_lvl)
gbm = GradientBoostingRegressor(n_estimators=150, max_depth=7,random_state=0, loss='ls', verbose=verbose_lvl)

models.extend([lr,rd,rf,gbm])

#cross-validate models using neg-squared-error scoring
scoring='neg_mean_squared_error'
target_df = df_train_raw[target_col]
print("Lauch cross validation")
for model in models:
    train_model(model,df_train_final,target_df,num_procs,eval_metric,cv_std)
    print_summary(model,eval_metric,cv_std)


# In[358]:


#construct dataframe from cross-validation results for plotting

eval_compa = pd.DataFrame([eval_metric, cv_std]).T
eval_compa.columns = ['mse_mean','mse_std']
eval_compa.index = ['Linear Regression','Ridge','Random Forest',
                   'Gradient Boosting']
eval_compa['Model Type'] = ["Linear","Linear","Non-Linear","Non-Linear"]
colors = {"Linear":"#273c75","Non-Linear": "#44bd32" }


# In[359]:


#Plot comparison of models
from matplotlib.patches import Patch

plt.figure(figsize=(10, 5))
eval_compa_sorted= eval_compa.sort_values('mse_mean',ascending=True)
eval_compa_sorted['mse_mean'].plot(kind = 'barh',
                                   color= eval_compa_sorted['Model Type'].replace(colors)).legend(
    [
        Patch(facecolor=colors['Linear']),
        Patch(facecolor=colors['Non-Linear'])
    ], ["Linear", "Non-Linear"])
plt.title("MSE score comparison")
plt.xlabel("MSE scores")
plt.savefig('models_mse_compa.png', bbox_inches = 'tight')
plt.show()


# **Now that the performance of the selected models has been evaluated, the best performing one, in this case the one boasting the lowest MSE, is used to train on the entire training dataset.**

# In[364]:


#select model with lowest MSE
model = min(eval_metric,key=eval_metric.get)
print(model)

#train model
model.fit(df_train_final,target_df)


# In[365]:


#saving model with pickle
from pickle import dump
from pickle import load

filename = 'finalize_model.sav'
dump(model, open(filename, 'wb'))


# In[366]:


#make predictions on training dataset with model
Pred_GBT = model.predict(df_train_final)


# In[371]:


# Gradient Boosting Model evaluation using visualisation on training dataset
ax1=sns.kdeplot(df_train_raw['salary'],color='b',label="Actuals")
ax1=sns.kdeplot(Pred_GBT,color='r',label="Fitted")
plt.legend()
plt.title("Actual vs Fitted Values Salary Prediction (Gradient Boosting Model)")
plt.xlabel("salary")
plt.ylabel("density")
plt.savefig('GradientB_Training_ActualsVsFitted.png')
plt.show()


# In[373]:


#store feature importances of Gradient Boosting model
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
    #linear models don't have feature_importances_
    importances = [0]*len(feature_df.columns)
    
feature_importances = pd.DataFrame({'feature':df_train_final.columns, 'importance':importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
#set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)
feature_importances.to_csv('feature_importances.csv')


# In[374]:


#Bar chart for top 10 feature importances
feature_importances[0:10].plot.bar(figsize=(20,10))
plt.savefig('feature_importances.png',bbox_inches = 'tight')
plt.show()


# In[375]:


feature_importances[0:10]


# ## Deployment of model

# In[ ]:


#Make predictions on new data i.e the test dataset, and save results
predictions = model.predict(df_test_final)
np.savetxt('predictions.csv', predictions, delimiter=',')


# In[383]:


#predictions = pd.Series(predictions)
predictions.head(20)


# In[386]:


#predictions made from loaded model, saved initially with Pickle
loaded_model = load(open(filename, 'rb'))
result = loaded_model.predict(df_test_final)
result[:20]


# In[ ]:




