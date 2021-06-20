# How-to-Handle-Missing-Data-with-Python
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
