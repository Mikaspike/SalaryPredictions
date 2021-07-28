# Predict salary for new job postings



### Introduction

This project's goal is to train and deploy a machine-learning model to predict salaries of new job postings.  What is the average salary for an employee from the oil industry, in an entry-level position and having majored in Engineering?  Or what is the salary range for downtown-based web companies?  And is work-experience a significant factor in determining salaries accross all industries?  Using exploratory data analysis on the given training dataset, we will be able to provide valuable insights to those kind of questions.  That process will enable us to determine the key features driving salaries, build our machine learning model and deploy it to make predictions on the job postings from the test dataset.


### Data

We have three csv files in the 'Data' folder of this repository:

- 'train_features': training dataset 
- 'train_salaries': contains the target variable 'salary' of each job ID from the training dataset  
- 'test_features': test dataset with the same features as the training dataset

### Other repository contents


'images' folder : contains png files of the various plots used in the Jupyter notebooks.

Jupyter notebooks:

- [SalaryPredictions_EDA](https://github.com/Mikaspike/SalaryPredictions/blob/main/SalaryPredictions_EDA.ipynb): contains the code for data cleansing, EDA and baseline creation.

- [SalaryPredictions_Models](https://github.com/Mikaspike/SalaryPredictions/blob/main/SalaryPredictions_Models.ipynb): contains the code for model evaluation, training and deployment

'predictions.csv' : the salaries predicted by the model from the test dataset


### Data preparation and cleansing

----

1. Data Loading into pandas dataframes

First five rows of training dataframe:

|    | jobId            | companyId   | jobType        | degree      | major     | industry   |   yearsExperience |   milesFromMetropolis |
|----|------------------|-------------|----------------|-------------|-----------|------------|-------------------|-----------------------|
|  0 | JOB1362684407687 | COMP37      | CFO            | MASTERS     | MATH      | HEALTH     |                10 |                    83 |
|  1 | JOB1362684407688 | COMP19      | CEO            | HIGH_SCHOOL | NONE      | WEB        |                 3 |                    73 |
|  2 | JOB1362684407689 | COMP52      | VICE_PRESIDENT | DOCTORAL    | PHYSICS   | HEALTH     |                10 |                    38 |
|  3 | JOB1362684407690 | COMP38      | MANAGER        | DOCTORAL    | CHEMISTRY | AUTO       |                 8 |                    17 |
|  4 | JOB1362684407691 | COMP7       | VICE_PRESIDENT | BACHELORS   | PHYSICS   | FINANCE    |                 8 |                    16 |


2. Merging train_features and train_salaries to form a single dataframe

3. Cleansing:
    * checking for missing data like nulls
        - no nulls were found in the datasets provided 
    * checking for invalid/redundant data 
        - jobID column was dropped from the training dataframe as it had a unique value for each row.
    * Removing duplicates
        - some duplicates were found and removed from the training set
    * Outlier check and treatment of data
        - some lower-bound outliers (salary of '0') were found and removed

|        | companyId   | jobType        | degree      | major       | industry   |   yearsExperience |   milesFromMetropolis |   salary |
|--------|-------------|----------------|-------------|-------------|------------|-------------------|-----------------------|----------|
|  30559 | COMP44      | JUNIOR         | DOCTORAL    | MATH        | AUTO       |                11 |                     7 |        0 |
| 495984 | COMP34      | JUNIOR         | NONE        | NONE        | OIL        |                 1 |                    25 |        0 |
| 652076 | COMP25      | CTO            | HIGH_SCHOOL | NONE        | AUTO       |                 6 |                    60 |        0 |
| 816129 | COMP42      | MANAGER        | DOCTORAL    | ENGINEERING | FINANCE    |                18 |                     6 |        0 |
| 828156 | COMP40      | VICE_PRESIDENT | MASTERS     | ENGINEERING | WEB        |                 3 |                    29 |        0 |




### Exploratory Data Analysis

----

For the amount of data points (1000000), it is expected for the target variable 'salary' to have a normal distribution.  This is the case as we can see from this plot:

![salary_distribution](/images/target_variable.png)*Salary distribution*


The features were splitted into separate lists based on their data types i.e categorical or numerical

Numerical list : yearsExperience, milesFromMetropolis, salary

Categorical list : companyId, jobType, degree, major, industry

Those lists are used to plot the relationship of each feature with the target variable for our _univariate analysis_.

**The follwing 1 x 2 plots are to be interpreted as the left, being the distribution of samples on the feature and the right, the dependance of the target variable on the feature.**

![years_Experience](/images/yearsExperience.png)*Years Experience*

* There is a clear positive correlation with Salary, i.e the more years in experience, the higher the salary



![miles](/images/milesFromMetropolis.png)*Miles from metropolis*

* We observe a negative correlation with Salaries. Typically the further away you are from city centre, the lower the salary.



![companyId](/images/companyId.png)*Company ID*

* CompanyID shows no correlation with salaries as it has a flat curve meaning most companies offer the same average salaries.



![jobType](/images/jobType.png)*Job Type*

* jobType shows a positive correlation with Salary. The higher the position, the higher the salary.


![Degrees](/images/degree.png)*Degrees*

* Degrees and salaries are also positively correlated.  The more advanced the degree, the higher the salary.



![Majors](/images/major.png)*Majors*

* Engineering, Business and Math are those commanding the highest salaries.



![Industry](/images/industry.png)*Industries*

* Oil, Finance and Web are those industries commanding the highest salaries.



#### Multivariate Analysis


![Correlation_matrix](/images/correlation_matrix.png)*Correlation matrix*


*  From the correlation heatmap , we can see that jobtypes, degree, major and yearsExperience have the highest correlation with Salary, in that order. Degree and major are strongly correlated. And we see the negative correlation of milesfromMetropolis with salary, as well as the near-zero correlation of Company ID with salary.  
* The colinearalities observed are not significant enough to require feature engineering processes such as dimensionality reduction or feature combination.  


### Baseline 

---

For our model performance baseline, a linear regression model using negative MeanSquaredError(MSE) scoring, has been selected.  The average MSE score using a five-fold cross-validation on the training dataset is 399.28.  

Our goal is to train and deploy a model boasting an MSE score of less than 360.



### Hypothesizing solutions

---

Now that we have established our baseline outcome, we proceed with looking at how the results can be improved.

From the EDA, we have seen that the categorical variables are the ones with the most significant correlation with 'salary'. Applying feature engineering on those variables such as label or one-hot encoding would definitely improve the predictive performance of the subsequent selected models.

We can also normalize the numerical variables by applying the appropriate scaling - this is generally good for linear models.

We will evaluate using MSE scoring, 2 linear and 2 non-linear models:

**Linear models** 

* Linear Regression
* Ridge Regression

**Non-linear models** 

* Random Forest
* Gradient Boosting

This will enable us to know what type of algorythm works better on our training dataset. The best performing one, will be the selected model for training and deployment.


### Feature Engineering

---

As mentioned, a normalizing process has been applied on the numerical variables of the training and test datasets.  The scaling algorythm used is Scikitlearn's MinMax scaler.  Here is a snapshot of the training set after normalizing:

|    |   index | companyId   | jobType   | degree      | major   | industry   |   yearsExperience |   milesFromMetropolis |   salary |
|----|---------|-------------|-----------|-------------|---------|------------|-------------------|-----------------------|----------|
|  0 |  753747 | COMP29      | MANAGER   | MASTERS     | COMPSCI | WEB        |          0.833333 |              0.323232 |      144 |
|  1 |  933220 | COMP27      | CEO       | MASTERS     | BIOLOGY | FINANCE    |          0.875    |              0.949495 |      122 |
|  2 |  783624 | COMP32      | CTO       | HIGH_SCHOOL | NONE    | HEALTH     |          1        |              0.474747 |      106 |
|  3 |  831054 | COMP40      | SENIOR    | DOCTORAL    | BIOLOGY | AUTO       |          1        |              0.707071 |      132 |
|  4 |  587715 | COMP51      | CFO       | BACHELORS   | BIOLOGY | WEB        |          0.375    |              0.565657 |       91 |


Next is a snapshot of the training set after undergoing one-hot encoding:



|    |   companyId_COMP0 |   companyId_COMP1 |   companyId_COMP10 |   companyId_COMP11 |   companyId_COMP12 |   companyId_COMP13 |   companyId_COMP14 |   companyId_COMP15 |   companyId_COMP16 |   companyId_COMP17 |   companyId_COMP18 |   companyId_COMP19 |   companyId_COMP2 |   companyId_COMP20 |   companyId_COMP21 |   companyId_COMP22 |   companyId_COMP23 |   companyId_COMP24 |   companyId_COMP25 |   companyId_COMP26 |   companyId_COMP27 |   companyId_COMP28 |   companyId_COMP29 |   companyId_COMP3 |   companyId_COMP30 |   companyId_COMP31 |   companyId_COMP32 |   companyId_COMP33 |   companyId_COMP34 |   companyId_COMP35 |   companyId_COMP36 |   companyId_COMP37 |   companyId_COMP38 |   companyId_COMP39 |   companyId_COMP4 |   companyId_COMP40 |   companyId_COMP41 |   companyId_COMP42 |   companyId_COMP43 |   companyId_COMP44 |   companyId_COMP45 |   companyId_COMP46 |   companyId_COMP47 |   companyId_COMP48 |   companyId_COMP49 |   companyId_COMP5 |   companyId_COMP50 |   companyId_COMP51 |   companyId_COMP52 |   companyId_COMP53 |   companyId_COMP54 |   companyId_COMP55 |   companyId_COMP56 |   companyId_COMP57 |   companyId_COMP58 |   companyId_COMP59 |   companyId_COMP6 |   companyId_COMP60 |   companyId_COMP61 |   companyId_COMP62 |   companyId_COMP7 |   companyId_COMP8 |   companyId_COMP9 |   jobType_CEO |   jobType_CFO |   jobType_CTO |   jobType_JANITOR |   jobType_JUNIOR |   jobType_MANAGER |   jobType_SENIOR |   jobType_VICE_PRESIDENT |   degree_BACHELORS |   degree_DOCTORAL |   degree_HIGH_SCHOOL |   degree_MASTERS |   degree_NONE |   major_BIOLOGY |   major_BUSINESS |   major_CHEMISTRY |   major_COMPSCI |   major_ENGINEERING |   major_LITERATURE |   major_MATH |   major_NONE |   major_PHYSICS |   industry_AUTO |   industry_EDUCATION |   industry_FINANCE |   industry_HEALTH |   industry_OIL |   industry_SERVICE |   industry_WEB |   yearsExperience |   milesFromMetropolis |
|----|-------------------|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------|--------------------|--------------------|--------------------|-------------------|-------------------|-------------------|---------------|---------------|---------------|-------------------|------------------|-------------------|------------------|--------------------------|--------------------|-------------------|----------------------|------------------|---------------|-----------------|------------------|-------------------|-----------------|---------------------|--------------------|--------------|--------------|-----------------|-----------------|----------------------|--------------------|-------------------|----------------|--------------------|----------------|-------------------|-----------------------|
|  0 |                 0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  1 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                 0 |                 0 |                 0 |             0 |             0 |             0 |                 0 |                0 |                 1 |                0 |                        0 |                  0 |                 0 |                    0 |                1 |             0 |               0 |                0 |                 0 |               1 |                   0 |                  0 |            0 |            0 |               0 |               0 |                    0 |                  0 |                 0 |              0 |                  0 |              1 |          0.833333 |              0.323232 |
|  1 |                 0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  1 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                 0 |                 0 |                 0 |             1 |             0 |             0 |                 0 |                0 |                 0 |                0 |                        0 |                  0 |                 0 |                    0 |                1 |             0 |               1 |                0 |                 0 |               0 |                   0 |                  0 |            0 |            0 |               0 |               0 |                    0 |                  1 |                 0 |              0 |                  0 |              0 |          0.875    |              0.949495 |
|  2 |                 0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  1 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                 0 |                 0 |                 0 |             0 |             0 |             1 |                 0 |                0 |                 0 |                0 |                        0 |                  0 |                 0 |                    1 |                0 |             0 |               0 |                0 |                 0 |               0 |                   0 |                  0 |            0 |            1 |               0 |               0 |                    0 |                  0 |                 1 |              0 |                  0 |              0 |          1        |              0.474747 |
|  3 |                 0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  1 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                 0 |                 0 |                 0 |             0 |             0 |             0 |                 0 |                0 |                 0 |                1 |                        0 |                  0 |                 1 |                    0 |                0 |             0 |               1 |                0 |                 0 |               0 |                   0 |                  0 |            0 |            0 |               0 |               1 |                    0 |                  0 |                 0 |              0 |                  0 |              0 |          1        |              0.707071 |
|  4 |                 0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  1 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |                 0 |                  0 |                  0 |                  0 |                 0 |                 0 |                 0 |             0 |             1 |             0 |                 0 |                0 |                 0 |                0 |                        0 |                  1 |                 0 |                    0 |                0 |             0 |               1 |                0 |                 0 |               0 |                   0 |                  0 |            0 |            0 |               0 |               0 |                    0 |                  0 |                 0 |              0 |                  0 |              1 |          0.375    |              0.565657 |


### Model selection and training

---


The training dataset is now ready to be used for evaluating the selected models.  As with our baseline metric, a 5-fold cross-validation with negative MSE scoring has been performed on each of the models.  The comparative results are summarized in the table below:


| Model                       |  mean mse  |    std    |
|:----------------------------|-----------:|----------:|
| LinearRegression            |    384.492 |  0.722863 |
| Ridge                       |    384.491 |  0.722655 |
| RandomForestRegressor       |    365.88  |  0.582813 |
| GradientBoostingRegressor   |    356.913 |  0.58563  |


Clearly the non-linear models have performed better with GradientBoosting scoring the lowest neg-MSE.


![MSE-Compa](/images/models_mse_compa.png)*Neg-MSE comparison*


Furthermore the GradientBoosting model is the only one to have enabled us to reach our goal of achieving an MSE score < 360 so its selection as our final model is made obvious.  After being fitted on the training dataset, the trained model is firstly saved using **Pickle**, then used to make predictions.    

'''python

from pickle import dump
from pickle import load

filename = 'finalize_model.sav'
dump(model, open(filename, 'wb'))

'''

A plot to visualise actual salaries versus predicted. 

![GradientB](/images/GradientB_Training_ActualsVsFitted.png)*ActualsVsFitted*


A useful feature of tree-based models like GradientBoosting is Feature_importances.  It allows us to see which features have been considered most important for the model to calculate its predictions.  Here is a bar-chart summarising the top 10 features of our model:

![FeatureImportances](/images/feature_importances.png)*Feature Importances*


### Deployment

---

Finally we can deploy our model by making salary predictions on unused data, i.e the test dataset.  The predictions made have been saved in the file 'predictions.csv' which can be found in the repository.  For future job description data, we can load our saved model using Pickle once more, and used it to predict salaries.

'''python

#predictions made from loaded model, saved initially with Pickle
loaded_model = load(open(filename, 'rb'))
result = loaded_model.predict(df_test_final)

'''





