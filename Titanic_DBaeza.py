# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:36:27 2018

@author: gring
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns

sns.set(style = 'darkgrid')

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Exploratory data analysis

## Column names present in each dataset
print(train_data.columns)

print(test_data.columns)

## Comparison of variables present
print(train_data.columns.difference(test_data.columns)) # Survived not in test
test_data['Survived'] = np.nan

all_data = train_data.append(test_data, ignore_index=True)

## Datatypes of the DataFrame
print(train_data.info())
print(test_data.info())

## Survival rates by Sex
sns.factorplot(x = 'Sex', hue = 'Survived',
               data = all_data, kind = 'count')
# Show low survival rates for men and higher survival rates for women

## Effect of passenger class on survival rates
sns.factorplot(x = 'Sex', hue = 'Pclass', col = 'Survived',
               data = all_data, kind = 'count',
               aspect = 0.7)
# Increased death for third class passengers

## Grouping of ages from 0 to teenager, teenager to adult, adult to
## elderly and above
all_data['Age_Range'] = pd.cut(all_data['Age'],
                                 bins = [0, 12, 18, 65, 100])

## Effect of passenger class on survival rates
sns.factorplot(x = 'Sex', hue = 'Age_Range', col = 'Survived',
               data = all_data, kind = 'count',
               aspect = 0.7)
# High relevance of Age Range on survival rates

## Effect of port of embarkation on survival rates
sns.factorplot(x = 'Sex', hue = 'Embarked', col = 'Survived',
               data = all_data, kind = 'count',
               aspect = 0.7)
# Same distribution for survived and perished, could imply lack of relevance
# for initial analysis

## Effect of number of siblings/spouses on survival rates
sns.factorplot(x = 'Sex', hue = 'SibSp', col  = 'Survived',
               data = all_data, kind = 'count',
               aspect = 0.7)

## Excessive amount of ticket fare values suggests creating ranges to
## ease data wrangling
print(len(all_data['Fare'].unique()))

all_data['Fare_Range'] = pd.qcut(all_data['Age'],
                                 q = np.linspace(0, 1, num = 11,
                                                    endpoint = True))

## Effect of ticket fare range on survival rates
sns.factorplot(x = 'Sex', hue = 'Fare_Range', col  = 'Survived',
               data = all_data, kind = 'count',
               aspect = 0.7)
# Cheap tickets are of high survival rates in men




## Missing data
### Missing data count
train_data.columns.values[train_data.isnull().sum() > 0]
# In Age (float64), Cabin (object) and Embarked (object) variables in
# training set
test_data.columns.values[test_data.isnull().sum() > 0]
# In Age (float64), Fare (float64) and Cabin (object in test set
