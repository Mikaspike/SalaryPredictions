#!/usr/bin/env python
# coding: utf-8

# # Salary Predictions Based on Job Descriptions

# We want to be able to predict salaries for new job postings by using a machine learning model.  In this notebook, we will clean the data first to be able to do an effective exploratory data analysis. 
# The purpose of which is to determine what kind of models to use and if any data preprocessing is required. 

# In[2]:


#Author
__author__ = "Michael Tin"
__email__ = "michael.tslin@gmail.com"


# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# **Functions**

# In[3]:


#Loading data
def load_file(file):
    '''loads .csv file to pandas dataframe'''
    return pd.read_csv(file)

#checking for null values
def check_null(df):
    '''returns a boolean for each variable in the dataframe'''
    return df.isnull().any()

#checking for duplicates
def check_duplicates(df):
    '''returns the duplicates from the dataframe'''
    return df[df.duplicated()]

#merging two dataframes by joining on common key      
def merge_data(df1, df2, key=None, left_index=False, right_index=False):
    '''perform inner join to return only records that are present in both dataframes'''
    return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)

#calculating lower and upper bounds for outliers using IQR rule (on a single numrical variable)
def outlier_calc(df,target):
    '''Interquartile rule of estimating outliers using stats from pandas describe() '''
    stat = df[target].describe()
    print(stat)
    IQR = stat['75%'] - stat['25%']
    upper = stat['75%'] + 1.5 * IQR
    lower = stat['25%'] - 1.5 * IQR
    return print('The upper and lower bounds for suspected outliers are {} and {}.'.format(upper, lower))

#list numerical and categorical variables
def num_cat(df):
    '''splitting numerical and categorical variables in seperate lists
    cat_list and num_list should be created prior to running this function'''
    for col in df:
        if is_numeric_dtype(df[col]):
            num_list.append(col)
        elif is_string_dtype(df[col]):
            cat_list.append(col)
            
def encode_label(df, col,target_var):
    '''encode the categories using average target variable(numerical) for each category to 
    replace category label'''
    cat_dict ={}
    #ensure that your categorical variables have been converted to type 'category' first
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df_train[df_train[col] == cat][target_var].mean()   
    df[col] = df[col].map(cat_dict)

def plot_var(df, col,target_var):
    '''
    Make plot for each features (col)
    left, the distribution of samples on the feature
    right, the dependance of the target variable on the feature
    '''
    plt.figure(figsize = (14, 6))
    plt.subplot(1, 2, 1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        #change the categorical variable to category type and order their level by the mean salary
        #in each category
        mean = df.groupby(col)[target_var].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1, 2, 2)
        #add categorical variable if it has too many categories to box plot
    if df[col].dtype == 'int64' or col == 'companyId':
        #plot the mean salary for each category and fill between the (mean - std, mean + std)
        mean = df.groupby(col)[target_var].mean()
        std = df.groupby(col)[target_var].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values + std.values,                          alpha = 0.1)
    else:
        sns.boxplot(x = col, y = target_var, data=df)
    
    plt.xticks(rotation=45)
    #label your y-axis according to your target variable
    plt.ylabel('Salaries')
    plt.tight_layout()
    plt.savefig(str(col)+'.png')
    plt.show()

    


# In[4]:


#Loading data
df_train = load_file('data/train_features.csv')
df_salaries = load_file('data/train_salaries.csv')
df_test = load_file('data/test_features.csv')


# ### Cleaning the data 
# 
# - Checking for missing data like nulls and treating if required
# - checking for invalid/redundant data
# - Removing duplicates
# - Outlier check and treatment of data
# 

# In[8]:


df_train.head()


# In[12]:


df_salaries.head()


# In[13]:


print(df_salaries.shape)
print(df_train.shape)


# In[5]:


#Same no of rows so we can merge the two datasets as well by joining on the JOBID column 
df_train = merge_data(df_train, df_salaries, key='jobId', left_index=False, right_index=False)
df_train.shape


# In[15]:


#checking for null values
check_null(df_train)


# In[16]:


check_null(df_test)


# No null values from both the training and test dataset so we do not need to process the data in that regard.

# In[9]:


#looking at unique value groups in data
df_train.iloc[:,:].nunique(dropna=False)


# 'jobID' only has unique values so we will remove it as it will bring no predicitve value for our models.

# In[6]:


df_train=df_train.drop(columns=['jobId'])


# In[7]:


df_train.info()


# In[12]:


#Duplicate check
check_duplicates(df_train)


# 186 duplicates out of 1000000 is not significant but we can still remove them

# In[8]:


#remove duplicate rows from our training dataset
duplic = check_duplicates(df_train)
df_train.drop(duplic.index, inplace=True)
df_train.shape


# In[17]:


#checking for outliers
target_var = 'salary'
outlier_calc(df_train,target_var)


# In[9]:


#examining lower bound outliers
df_train[df_train.salary < 8.5]


# In[16]:


#check potential outlier above upper bound
df_train.loc[df_train.salary > 220.5, 'jobType'].value_counts()


# In[17]:


#check 'Junior' outliers
df_train[(df_train.salary > 220.5) & (df_train.jobType == 'JUNIOR')]


# The above-upper bound outliers are valid ones since they are records with jobytpes belonging to high-paying industries such as Oil and Finance.  Even 'juniors' are well paid compared to others.
# 
# The below-lower bound outliers all have salary of 0. Those entries appear to be legitimate corrupt data as there is no valid information to suggest as why a salary could be 0, e.g volunteering.  So we can remove them

# In[10]:


# remove those below-lower bound outliers from our training dataset
df_train = df_train[df_train.salary > 8.5]


# ### Exploratory Data Analysis

# In[17]:


#Analysing the target variable's distribution.
plt.figure(figsize = (12, 6))
plt.subplot(1,2,1)
sns.boxplot(df_train.salary)
plt.subplot(1,2,2)
sns.distplot(df_train.salary, bins=20)
plt.savefig('target_variable.png')
plt.show()


# We observe a normal distribution for 'Salary'

# In[12]:


#classifying numerical and categorical variables
num_list=[]
cat_list=[]
num_cat(df_train)


# In[13]:


print("Numerical variables:")
print(*num_list, sep =", ")
print("categorical variables:")
print(*cat_list, sep =", ")


# ### Univariate analysis by plotting the relationship of each feature with Salary

# In[31]:


plot_var(df_train,'yearsExperience',target_var)


# **Clear positive correlation with Salary.  The more years in job experience, the higher the salary**

# In[32]:


plot_var(df_train,'milesFromMetropolis',target_var)


#  **We observe a negative correlation with Salaries.  Typically the further away you are from city centre, the lower the salary.**

# In[26]:


#plotting all the categorical variables
for col in df_train[cat_list]:
    plot_var(df_train,col,target_var)


# - CompanyID shows no correlation with salaries as it has a flat curve meaning all companies have the same avegrage salaries
# 
# - jobType shows a positive correlation with Salary. The higher the position, the higher the salary
# 
# - Degrees and salaries are also positively correlated.  The more advanced the degree, the higher the salary
# 
# - Major - Engineering, Business and Math are those commanding the highest salaries
#  
# - Industry - Oil, Finance and Web are thse industries commanding the highest salaries

# # Multivariate Analysis with correlation
# 
# We proceed with multivariate analysis to look for correlation between variables and not just with the target variable.

# In[27]:


#Encoding categorical variables to average salary per category
for col in cat_list:
     encode_label(df_train, col,target_var)
print (df_train[cat_list].head())


# In[37]:


#convert encoded column catetories to int to make the correlation matrix work with all variables
for x in df_train.columns:
    if df_train[x].dtype.name == "category":
        df_train[x] = df_train[x].astype('int64')


# In[38]:


df_train.info()


# In[39]:


#check correlation of numerical features

fig = plt.figure(figsize=(12, 10))
features = ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
sns.heatmap(df_train[features + ['salary']].corr(), cmap='Blues', annot=True)
plt.xticks(rotation=45)
plt.savefig('correlation_matrix.png')
plt.show()


# **Observations**
# 
# We can see that jobtypes, degree, major and yearsExperience have the highest correlation with Salary, in that order.  Degree and major are strongly correlated.
# 
# And we see the negative correlation of milesfromMetropolis with salary, as well as the near-zero correlation of Company ID with salary.  
# 

# ###  Establishing a baseline

# In[29]:


#We will use a very simple model to establish a baseline - measuring Root Mean Squared Error
#by simply measuring the actual salary against the average salary per industry.

#merging average mean salary per industry with our training dataframe
df_avg_ind = pd.DataFrame(df_train.groupby('industry')['salary'].mean())
df_base = df_train.merge(df_avg_ind,on='industry',how='inner')


# In[54]:


df_base.head(10)


# In[47]:


#  root mean squared error function or we could have use the function from scikit learn
# from sklearn.metrics import mean_squared_error 

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# In[50]:


from math import sqrt
#calculating the root MSE for the average salary per industry

rmse = round(rmse_metric(df_base.salary_x,df_base.salary_y),3)
print("Baseline root MSE", rmse)


# Our baseline root MSE is excessively high so we can consider this metric as too unrealistic. Instead we will establish a more realistic baseline using neg MSE scoring on a five-fold cross-validation of our training dataset, with Linear Regression.

# In[55]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
array = df_base.values
#splitting the new df in input and target variables
X = array[:,0:7]
Y = array[:,7]

neg_mse = 'neg_mean_squared_error'
results = cross_val_score(model,X,Y,cv=5,scoring=neg_mse)
mean_mse = -1*np.mean(results)
std_mse = np.std(results)
print('Average MSE:\n', mean_mse)
print('Standard deviation during CV:\n', std_mse)


# **Therefore the baseline metric is an MSE score of 399.  We will aim to get this score lower than 360.** 

# ### Hypothesizing  solutions

# Now that we have established our baseline model, we proceed with looking at how the results can be improved. 
# 
# From the EDA, we have seen that the categorical variables are the ones with the most significant correlation with 'salary'.  Applying feature engineering on those variables such as label or one-hot encoding would definitely improve the predictive performance of the subsequent selected models.  
# 
# We can also normalize the numerical variables by applying the appropriate scaling - this is generally good for linear models.  
# 
# 
# We will evaluate using MSE scoring, 2 linear and 2 non-linear models:
# 
# Linear models -
# 
# - Linear Regression
# - Ridge Regression
# 
# Non-linear models -
# 
# - Random Forest
# - Gradient Boosting
# 
# This will enable us to know what type of algorythm works better on our training dataset.  The best performing one, will be the selected model for training and deployment.
# 

# ### The modelling process can be viewed in the notebook SalaryPredictions_Models
