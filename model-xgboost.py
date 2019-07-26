import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# Getting the data
train = pd.read_csv('train.csv')

train = train[train.GrLivArea < 4500]

# train_label = np.log1p(train['SalePrice'])
train_label = train['SalePrice']

test = pd.read_csv('test.csv')
test = pd.merge(test, pd.read_csv('sample_submission.csv'), on='Id', how='left')

newdf = pd.concat([train, test], axis=0).reset_index(drop=True)

newdf['SalePrice'] = np.log1p(newdf['SalePrice'])

# Filling the numericals columns with 0
fill0_cols = ['BsmtFinSF1',
            'BsmtFinSF2',
            'BsmtFullBath',
            'BsmtHalfBath',
            'BsmtUnfSF',
            'GarageArea',
            'GarageCars',
            'LotFrontage',
            'MasVnrArea',
            'TotalBsmtSF']
for col in fill0_cols:
    newdf[col] = newdf[col].fillna(0)

# Filling the non-numerical columns with 'None'
fillnone_cols = ['BsmtCond',
                'BsmtExposure',
                'BsmtFinType1',
                'BsmtFinType2',
                'BsmtQual',
                'Fence',
                'FireplaceQu',
                'GarageCond',
                'GarageFinish',
                'GarageQual',
                'GarageType',
                'MasVnrType',
                'MiscFeature',
                'PoolQC',
                'Alley']
for col in fillnone_cols:
    newdf[col] = newdf[col].fillna('None')

# Filling the others columns with treir most commun value
newdf['Electrical'] = newdf['Electrical'].fillna('SKbkr')
newdf['Exterior1st'] = newdf['Exterior1st'].fillna('VinylSd')
newdf['Exterior2nd'] = newdf['Exterior2nd'].fillna('VinylSd')
newdf['Functional'] = newdf['Functional'].fillna('Typ')
newdf['KitchenQual'] = newdf['KitchenQual'].fillna('TA')
newdf['MSZoning'] = newdf['MSZoning'].fillna('RL')
newdf['SaleType'] = newdf['SaleType'].fillna('WD')

newdf['MSSubClass'] = newdf['MSSubClass'].apply(str)

# Feature Engineering
newdf['GarageYrBlt'] = newdf['GarageYrBlt'].apply(lambda x: 2007 if x == 2207 else x)
newdf['HouseStyle'] = newdf['HouseStyle'].apply(lambda x: '1Story' if (x == '1.5Fin') or (x == '1.5Unf') else '2Story' if (x == '2.5Fin') or (x == '2.5Unf') else x)
newdf['BsmtExposure'] = newdf['BsmtExposure'].apply(lambda x: 'None' if x == 'No' else x)
newdf['HasAlleyAcess'] = newdf['Alley'].apply(lambda x: 0 if x == 'None' else 1)
# newdf['BsmtFinSF'] = newdf['BsmtFinSF1'] + newdf['BsmtFinSF2']
newdf['TotalPorchSF'] = newdf['OpenPorchSF'] + newdf['EnclosedPorch'] + newdf['3SsnPorch'] + newdf['ScreenPorch'] + newdf['WoodDeckSF']
# newdf['HasPorch'] = newdf['TotalPorchSF'].apply(lambda x: 1 if x > 0 else 0)
newdf['HasMasVnr'] = newdf['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
# newdf['HasFireplace'] = newdf['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
newdf['Has2ndFlr'] = newdf['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
newdf['TotalHouseSF'] = newdf['2ndFlrSF'] + newdf['1stFlrSF'] + newdf['TotalBsmtSF']

mapping = {'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
           'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
           'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
           'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
           'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
           'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
           'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},
           'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
           'CentralAir': {'N': 0, 'Y': 1},
           'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
           'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
           'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},
           'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
           'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}}

bins = [-1] + [x for x in range(2, 111, 2)] + [np.inf]
labels = ['{}to{}'.format(x, x+2) for x in range(0, 110, 2)] + ['gt110']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update(LotFrontage=labs)

newdf['LotFrontage'] = pd.cut(newdf['LotFrontage'], bins, labels=labels)

bins = [x for x in range(1299, 17001, 100)] + [np.inf]
labels = ['{}to{}'.format(x, x+100) for x in range(1300, 17000, 100)] + ['gt17000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update(LotArea=labs)

newdf['LotArea'] = pd.cut(newdf['LotArea'], bins, labels=labels)

# bins = [-1] + [x for x in range(2, 651, 2)] + [np.inf]
# labels = ['{}to{}'.format(x, x+2) for x in range(0, 650, 2)] + ['gt650']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.update(BsmtFinSF1=labs)

# newdf['BsmtFinSF1'] = pd.cut(newdf['BsmtFinSF1'], bins, labels=labels)

# bins = [-1] + [x for x in range(2, 651, 2)] + [np.inf]
# labels = ['{}to{}'.format(x, x+2) for x in range(0, 650, 2)] + ['gt650']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.update(BsmtFinSF2=labs)

# newdf['BsmtFinSF2'] = pd.cut(newdf['BsmtFinSF2'], bins, labels=labels)

# bins = [-1] + [x for x in range(50, 1001, 50)] + [np.inf]
# labels = ['{}to{}'.format(x, x+50) for x in range(0, 1000, 50)] + ['gt1000']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.update(BsmtUnfSF=labs)

# newdf['BsmtUnfSF'] = pd.cut(newdf['BsmtUnfSF'], bins, labels=labels)

bins = [-1] + [x for x in range(50, 1250, 50)] + [np.inf]
labels = ['{}to{}'.format(x, x+40) for x in range(50, 1250, 50)] + ['gt1000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update(TotalBsmtSF=labs)

newdf['TotalBsmtSF'] = pd.cut(newdf['TotalBsmtSF'], bins, labels=labels)

bins = [x for x in range(300, 1801, 20)] + [np.inf]
labels = ["{}to{}".format(x, x+20) for x in range(300, 1800, 20)] + ['gt1800']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update({'1stFlrSF': labs})

newdf['1stFlrSF'] = pd.cut(newdf['1stFlrSF'], bins, labels=labels)

# bins = [x for x in range(300, 2301, 40)] + [np.inf]
# labels = ["{}to{}".format(x, x+40) for x in range(300, 2300, 40)] + ['gt2300']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.update(GrLivArea=labs)

# newdf['GrLivArea'] = pd.cut(newdf['GrLivArea'], bins, labels=labels)

# bins = [-1] + [x for x in range(40, 801, 40)] + [np.inf]
# labels = ["{}to{}".format(x, x+40) for x in range(0, 800, 40)] + ['gt800']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.update(GarageArea=labs)

# newdf['GarageArea'] = pd.cut(newdf['GarageArea'], bins, labels=labels)

# newdf['HasWoodDeck'] = newdf['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)

newdf = newdf.replace(mapping)

drop_cols = ['Utilities',
            'GarageYrBlt',
            'SalePrice',
            '2ndFlrSF',
            'MasVnrArea',
            'PoolQC',
            'Alley',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'Street',
            'MiscFeature',
            'PoolArea',
            'RoofMatl',
            'LowQualFinSF',
            'Heating',
            'MiscVal',
            'WoodDeckSF'
            ]

newdf.drop(drop_cols, axis=1, inplace=True)
# newdf.profile_report().to_file('profile_completo_af.html')
newdf = pd.get_dummies(newdf)
# newdf.columns = [x.replace(" ", "_") for x in newdf.columns.values]
newdf.to_csv('teste_file.csv', index=False)

trlen = len(train)

train = newdf[:trlen]

# X_train, X_test, y_train, y_test = train_test_split(train.drop('Id', axis=1), train_label, test_size=0.3, random_state=50)

# dtrain = xgb.DMatrix(X_train, y_train)
# dtest = xgb.DMatrix(X_test, y_test)
dtrain = xgb.DMatrix(train.drop('Id', axis=1), train_label)

test = newdf[trlen:]

# label = pd.read_csv('submission_test_top_top.csv')['SalePrice']#.apply(lambda x: math.log1p(x))
label = pd.read_csv('sample_submission.csv')['SalePrice']#.apply(lambda x: math.log1p(x))

dtest = xgb.DMatrix(test.drop('Id', axis=1), label)

def rmsle(pred: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    pred[pred < -1] = -1 + 1e-6
    error = float(np.sqrt(np.sum(np.power(np.log1p(y) - np.log1p(pred), 2)) / len(y)))
    return 'MyRMSLE', error

param = {'max_depth': 2, 'objective': 'reg:squarederror', 'disable_default_eval_metric': 1}
param['nthread'] = 4
# param['eval_metrics'] = 'rmse'

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round=150

xgb_model = xgb.train(param, dtrain, num_round, evallist, feval=rmsle)

test = newdf[trlen:]
id_test = newdf[trlen:]['Id']

label = pd.read_csv('sample_submission.csv')['SalePrice']#.apply(lambda x: math.log10(x))

dsub = xgb.DMatrix(test.drop('Id', axis=1), label)

# print(np.expm1(xgb_model.predict(dsub)))
print(xgb_model.predict(dsub))
sub = pd.DataFrame()
sub['Id'] = id_test
# sub['SalePrice'] = np.expm1(xgb_model.predict(dsub))
sub['SalePrice'] = xgb_model.predict(dsub)
sub.to_csv('submission_test.csv', index=False)

# # model = SelectKBest(score_func=f_regression, k=50)
# # model.fit_transform()
# bestfeatures = SelectKBest(score_func=f_regression, k=10)
# fit = bestfeatures.fit(train,train_label)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(train.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Features','Score']  #naming the dataframe columns
# featureScores.sort_values(by=['Score'], ascending=False).to_csv('scores.csv', index=False)

# colcats = newdf.select_dtypes(include=[np.object, 'category']).columns.values
# labenc = LabelEncoder()
# for i in colcats:
#     newdf[i] = labenc.fit_transform(newdf[i])
