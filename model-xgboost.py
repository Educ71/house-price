import pandas as pd
import xgboost as xgb
import numpy as np
import pandas_profiling

# Getting the data
train = pd.read_csv('train.csv')

train = train[train.GrLivArea < 4500]

train_label = train['SalePrice']

test = pd.read_csv('test.csv')
test = pd.merge(test, pd.read_csv('sample_submission.csv'), on='Id', how='left')

df = pd.concat([train, test], axis=0).reset_index(drop=True)

df['SalePrice'] = np.log1p(df['SalePrice'])

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
    df[col] = df[col].fillna(0)

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
    df[col] = df[col].fillna('None')

# Filling the others columns with treir most commun value
df['Electrical'] = df['Electrical'].fillna('SKbkr')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
df['Functional'] = df['Functional'].fillna('Typ')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['MSZoning'] = df['MSZoning'].fillna('RL')
df['SaleType'] = df['SaleType'].fillna('WD')

df['MSSubClass'] = df['MSSubClass'].apply(str)

# Feature Engineering
df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: 2007 if x == 2207 else x)
df['HouseStyle'] = df['HouseStyle'].apply(lambda x: '1Story' if (x == '1.5Fin') or (x == '1.5Unf') else '2Story' if (x == '2.5Fin') or (x == '2.5Unf') else x)
df['BsmtExposure'] = df['BsmtExposure'].apply(lambda x: 'None' if x == 'No' else x)
df['HasAlleyAcess'] = df['Alley'].apply(lambda x: 0 if x == 'None' else 1)
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
df['HasMasVnr'] = df['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has2ndFlr'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['TotalHouseSF'] = df['2ndFlrSF'] + df['1stFlrSF'] + df['TotalBsmtSF']

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

df['LotFrontage'] = pd.cut(df['LotFrontage'], bins, labels=labels)

bins = [x for x in range(1299, 17001, 100)] + [np.inf]
labels = ['{}to{}'.format(x, x+100) for x in range(1300, 17000, 100)] + ['gt17000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update(LotArea=labs)

df['LotArea'] = pd.cut(df['LotArea'], bins, labels=labels)

bins = [-1] + [x for x in range(50, 1250, 50)] + [np.inf]
labels = ['{}to{}'.format(x, x+40) for x in range(50, 1250, 50)] + ['gt1000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update(TotalBsmtSF=labs)

df['TotalBsmtSF'] = pd.cut(df['TotalBsmtSF'], bins, labels=labels)

bins = [x for x in range(300, 1801, 20)] + [np.inf]
labels = ["{}to{}".format(x, x+20) for x in range(300, 1800, 20)] + ['gt1800']

labs = {labels[value]: value for value in range(len(labels))}
mapping.update({'1stFlrSF': labs})

df['1stFlrSF'] = pd.cut(df['1stFlrSF'], bins, labels=labels)

df = df.replace(mapping)

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

df.drop(drop_cols, axis=1, inplace=True)
df = pd.get_dummies(df)
df.to_csv('teste_file.csv', index=False)

trlen = len(train)

train = df[:trlen]

dtrain = xgb.DMatrix(train.drop('Id', axis=1), train_label)

def rmsle(pred: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    pred[pred < -1] = -1 + 1e-6
    error = float(np.sqrt(np.sum(np.power(np.log1p(y) - np.log1p(pred), 2)) / len(y)))
    return 'MyRMSLE', error

param = {'max_depth': 2, 'objective': 'reg:squarederror', 'disable_default_eval_metric': 1}
param['nthread'] = 4

evallist = [(dtrain, 'train')]

num_round=150

xgb_model = xgb.train(param, dtrain, num_round, evallist, feval=rmsle)

test = df[trlen:]

dtest = xgb.DMatrix(test.drop('Id', axis=1))

# Saving the result in a file to submit to the competition
sub = pd.DataFrame()
sub['Id'] = test.Id
sub['SalePrice'] = xgb_model.predict(dtest)
sub.to_csv('submission_test.csv', index=False)
