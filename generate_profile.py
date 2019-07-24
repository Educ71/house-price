import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import numpy as np

# train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

test['SalePrice'] = pd.read_csv('sample_submission.csv')['SalePrice']

test['GarageYrBlt'] = test['GarageYrBlt'].apply(lambda x: 2007 if x == 2207 else x)
# cols = ['2ndFlrSF', 'LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2', 'MasVnrArea', 'BsmtUnfSF', 'BsmtHalfBath', 'GarageArea']
cols = test.select_dtypes(include=[np.number]).columns.values
test[cols] = test[cols].fillna(0)

bins = [-1, 20, 40, 60, 80, 100, np.inf]
labels = '0to19 20to39 40to59 60to79 80to99 gt100'.split()

test['LotFrontage'] = pd.cut(test['LotFrontage'], bins, labels=labels)

bins = [1000, 3000, 5000, 7000, 9000, 11000, np.inf]
labels = '1kto3k 3kto5k 5kto7k 7kto9k 9kto11k gt11k'.split()

test['LotArea'] = pd.cut(test['LotArea'], bins, labels=labels)

bins = [-1, 50, 100, 150, 200, np.inf]
labels = '0to50 51to100 101to150 151to200 gt200'.split()

test['MasVnrArea'] = pd.cut(test['MasVnrArea'], bins, labels=labels)

bins = [-1, 150, 300, 450, 600, np.inf]
labels = '0to150 151to300 301to450 451to600 gt600'.split()

test['BsmtFinSF'] = test['BsmtFinSF1'] + test['BsmtFinSF2']

test['BsmtFinSF'] = pd.cut(test['BsmtFinSF'], bins, labels=labels)

bins = [-1, 200, 400, 600, 800, 1000, np.inf]
labels = '0to200 201to400 401to600 601to800 801to1000 gt1000'.split()

test['BsmtUnfSF'] = pd.cut(test['BsmtUnfSF'], bins, labels=labels)

bins = [300, 800, 1300, 1800, np.inf]
labels = '300to800 801to1300 1301to1800 gt1800'.split()

test['1stFlrSF'] = pd.cut(test['1stFlrSF'], bins, labels=labels)

# test['Has2ndFlr'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

bins = [-1, 300, 600, 900, 1200, np.inf]
labels = '0to300 301to600 601to900 901to1200 gt1200'.split()

test['2ndFlrSF'] = pd.cut(test['2ndFlrSF'], bins, labels=labels)

bins = [300, 800, 1300, 1800, 2300, np.inf]
labels = '300to800 801to1300 1301to1800 1801to2300 gt2300'.split()

test['GrLivArea'] = pd.cut(test['GrLivArea'], bins, labels=labels)

bins = [-1, 200, 400, 600, 800, np.inf]
labels = '0to200 201to400 401to600 601to800 gt800'.split()

test['GarageArea'] = pd.cut(test['GarageArea'], bins, labels=labels)

bins = [-1, 50, 100, 150, 200, 250, np.inf]
labels = '0to50 51to100 101to150 151to200 201to250 gt250'.split()

test['WoodDeckSF'] = pd.cut(test['WoodDeckSF'], bins, labels=labels)

# train.profile_report().to_file('profile_train.html')

test.profile_report().to_file('profile_testinho.html')
