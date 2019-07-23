import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')

train_label = train['SalePrice']

train.drop('SalePrice', axis=1, inplace=True)

test = pd.read_csv('test.csv')

newdf = pd.concat([train, test], axis=0).reset_index(drop=True)

newdf['MSSubClass'] = newdf['MSSubClass'].apply(lambda x: 'SC{}'.format(x))

lb_enc = LabelEncoder()

col_list = ('LotShape LandSlope ExterQual ExterCond BsmtQual BsmtCond BsmtExposure HeatingQC CentralAir '
'KitchenQual FireplaceQu GarageFinish GarageQual GarageCond PavedDrive PoolQC').split()

newdf[col_list] = newdf[col_list].fillna('NA')

mapping = [{'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
            {'Gtl': 2, 'Mod': 1, 'Sev': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
            {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'N': 0, 'Y': 1},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
            {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
            {'Y': 2, 'P': 1, 'N': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}]

for i in range(len(col_list)):
    newdf[col_list[i]] = newdf[col_list[i]].map(mapping[i])



train = pd.get_dummies(train)

dtrain = xgb.DMatrix(train.drop('SalePrice', 1), train['SalePrice'])

test = pd.get_dummies(test)

label = pd.read_csv('sample_submission.csv')['SalePrice']

dtest = xgb.DMatrix(test, label)

param = {'max_depth': 2, 'objective': 'reg:squarederror'}
param['nthread'] = 4
param['eval_metric'] = ['rmse', 'mae']

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round=300

model = xgb.train(param, dtrain, num_round, evallist)

print(model.predict(dtest))