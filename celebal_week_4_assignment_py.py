# -*- coding: utf-8 -*-
"""Celebal_week_4_assignment.py

Original file is located at
    https://colab.research.google.com/drive/1aMtdxC9ydfhO8clBkfOd8QHVwIyYQJpw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/clv_data.csv')
data.head()

data.shape

# Checking for missing values:
print("\nMissing values in each column:\n", data.isnull().sum())

print("\nSummary Statistics:\n", data.describe(include='all'))

# Create the heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(data.isnull())

plt.title('Missing Values Heatmap')

plt.show()

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestRegressor

data_filled = data.copy()

numerical_cols = data_filled.select_dtypes(include=['number']).columns

categorical_cols = data_filled.select_dtypes(exclude=['number']).columns



# numerical features

imputer_num = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)

data_filled[numerical_cols] = imputer_num.fit_transform(data_filled[numerical_cols])



# Fill categorical features

for col in categorical_cols:

    data_filled[col] = data_filled[col].fillna(data_filled[col].mode()[0])



# Verifying the missing values now

print("\nMissing values after imputation:\n", data_filled.isnull().sum())

# Analyze 'days_on_platform'

print("\ndays_on_platform Count:\n",
data_filled['days_on_platform'].value_counts())

sns.countplot(x='days_on_platform', data=data_filled)

plt.title('days_on_platform Count')

plt.show()



# Analyze 'purchases'

print("\npurchases Distribution:\n",
data_filled['purchases'].value_counts())

sns.countplot(x='purchases', data=data_filled)

plt.title('purchases Distribution')

plt.show()



# Analyze 'age'

print("\nAge Statistics:\n", data_filled['age'].describe())

sns.histplot(x='age', data=data_filled, kde=True)

plt.title('Age Distribution')

plt.show()



# Analyze 'gender'

print("\ngender Distribution:\n",
data_filled['gender'].value_counts())

sns.countplot(x='gender', data=data_filled)

plt.title('gender Distribution')

plt.show()

# Boxplot for outlier detection
for age in data:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[age])
    plt.title("Boxplot of age")
plt.show()

# Analyze days_on_platform based on gender

print("\ndays_on_platform by Gender:\n", data_filled.groupby('gender')['days_on_platform'].value_counts())

sns.countplot(x='gender', hue='days_on_platform', data = data_filled)

plt.title('days_on_platform by Gender')

plt.show()



# Analyze days_on_platform based on purchases class

print("\ndays_on_platform by purchases :\n", data_filled.groupby('purchases')['days_on_platform'].value_counts())

sns.countplot(x='purchases', hue='days_on_platform', data = data_filled)

plt.title('days_on_platform by purchases')

plt.show()



# Analyze days_on_platform based on age groups (create age bins)

data_filled['age_group'] = pd.cut(data_filled['age'], bins=[0, 18, 65, 100], labels=['Child', 'Adult', 'Senior'])

print("\ndays_on_platform by Age Group:\n", data_filled.groupby('age_group')['days_on_platform'].value_counts())

sns.countplot(x='age_group', hue='days_on_platform', data = data_filled)

plt.title('days_on_platform by Age Group')

plt.show()



# Further exploration: Analyze days_on_platform based on combinations of factors

# Example: days_on_platform based on gender and class

print("\ndays_on_platform by Gender and Class:\n", data_filled.groupby(['gender', 'purchases'])['days_on_platform'].value_counts())

sns.catplot(x='gender', hue='days_on_platform', col='purchases', kind='count', data = data_filled)

plt.show()

# Create Age Group

bins = [0, 12, 18, 40, 60, 80]

labels = ['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior']

data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)



# Visualize Survival by Age Group

plt.figure(figsize=(10, 6))

sns.countplot(x='age_group', hue='days_on_platform', data = data, palette='magma')

plt.title('days_on_platform Based on Age Group')

plt.xlabel('Age Group')

plt.ylabel('Count')

plt.legend(title='days_on_platform', labels=['No', 'Yes'])

plt.show()

# Correlation heatmap

plt.figure(figsize=(12, 8))

# Select only numeric features for correlation calculation

numeric_features = data.select_dtypes(include=np.number)

sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")

plt.title('Feature Correlation Heatmap')

plt.show()
