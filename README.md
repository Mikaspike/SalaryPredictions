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

- [SalaryPredictions_EDA](/Mikaspike/SalaryPredictions/blob/main/SalaryPredictions_EDA.ipynb){:target="_blank"} : contains the code for data cleansing, EDA and baseline creation.

- [SalaryPredictions_Models](https://github.com/Mikaspike/SalaryPredictions/blob/main/SalaryPredictions_Models.ipynb){:target="_blank"} : contains the code for model evaluation, training and deployment

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

**The follwing 1 x 2 plots are to be interpresed as the left, being the distribution of samples on the feature and the right, the dependance of the target variable on the feature.**

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

For our model performance baseline, a linear regression model using negative MeanSquaredError(MSE) scoring, has been selected.  The average MSE score using a five-fold cross-validation on the training dataset is 399.28.  

Our goal is to train and deploy a model boasting an MSE score of less than 360.

---

### Hypothesizing solutions

Now that we have established our baseline outcome, we proceed with looking at how the results can be improved.

From the EDA, we have seen that the categorical variables are the ones with the most significant correlation with 'salary'. Applying feature engineering on those variables such as label or one-hot encoding would definitely improve the predictive performance of the subsequent selected models.

We can also normalize the numerical variables by applying the appropriate scaling - this is generally good for linear models.

We will evaluate using MSE scoring, 2 linear and 2 non-linear models:

**Linear models -

* Linear Regression
* Ridge Regression

**Non-linear models -

* Random Forest
* Gradient Boosting

This will enable us to know what type of algorythm works better on our training dataset. The best performing one, will be the selected model for training and deployment.
