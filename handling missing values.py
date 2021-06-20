"""
Missing values can occur for a variety of causes, including observations
that were not recorded or data corruption. There are algorithms that can be 
made to be robust to missing data, such as the k-Nearest Neighbors algorithm,
which can omit a column from a distance measure when a value is absent. When
creating a forecast, Naive Bayes can also take into account missing values.

Many machine learning methods do not accept data with missing values, so
 handling missing data is critical.
  The variable names are as follows:

0. Number of times pregnant.
1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
2. Diastolic blood pressure (mm Hg).
3. Triceps skinfold thickness (mm).
4. 2-Hour serum insulin (mu U/ml).
5. Body mass index (weight in kg/(height in m)^2).
6. Diabetes pedigree function.
7. Age (years).
8. Class variable (0 or 1).

#### Algorithms that Support Missing Values are:
     k-Nearest Neighbors that can ignore a column from a distance measure
           when a value is missing. 
     Naive Bayes can also support missing values when making a prediction.

"""
import gc
import os
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
##### reading data

path = 'D:/How to Handle Missing Data with Python/'
os.listdir(path)
df_train= pd.read_csv(path+ 'Diabetes.csv', header=None)
############################## 
df_train.head() # sample of the first 5 rows

'''
We know that zero isn't valid for those measurements; for example, a zero for BMI body mass index or blood pressure isn't valid.
'''
df_train.describe()

'''
There are columns with a minimum value of zero, as can be seen (0). A value of zero is illogical and indicates an incorrect or missing value.
'''
print((df_train==0).sum().sort_values(ascending=False))
'''
We observe that the column of ( Hour serum insulin(374),
                               Triceps skinfold thickness(227),
                               Number of times pregnant(111),
                               blood pressure(35), weight(11),
                               Plasma glucose concentration(5))
 have zero's values which do not make sense, so we will mark the zero's
 values by nan to ignore them
''' 
########### Mark Missing Values as nan ##############
from numpy import nan
df_train[[0,1,2,3,4,5]]=df_train[[0,1,2,3,4,5]].replace(0, nan)
print(df_train.isnull().sum().sort_values(ascending=False))
df_train.head()

'''
some an algorithm such as LinearDiscriminantAnalysis 
does not work when there are missing values in the dataset
'''
# 1 Remove Rows With Missing Values...shape_before(768, 9)
df_train.dropna(inplace=True)
df_train.shape # shape_after (336, 9)
# 2 Impute Missing Values (replace missing values)
'''
When replacing a missing value, we have a variety of possibilities to select,
such as:
         constant value, such as 0, that is different from all other values.
         A value from a different record that was chosen at random.
        The column's mean, median, or mode value.
'''
# manually impute missing values with the mean value
# class_mean=df_train[8].mean()
# column_0=df_train[0].mean()
mean_columns=df_train.mean()
df_train.fillna(mean_columns, inplace=True)
# a count of the number of missing values in each column
print(df_train.isnull().sum().sort_values(ascending=False))
