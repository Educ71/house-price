import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math

train = pd.read_csv('train.csv')

train_label = train['SalePrice']#.apply(lambda x: math.log10(x))

train.drop('SalePrice', axis=1, inplace=True)

test = pd.read_csv('test.csv')

newdf = pd.concat([train, test], axis=0).reset_index(drop=True)

newdf['MSSubClass'] = newdf['MSSubClass'].apply(lambda x: 'SC{}'.format(x))
newdf['GarageYrBlt'] = newdf['GarageYrBlt'].apply(lambda x: 2007 if x == 2207 else x)
cols = newdf.select_dtypes(include=[np.number]).columns.values
newdf[cols] = newdf[cols].fillna(0)

# bins = [-1, 20, 40, 60, 80, 100, np.inf]
# labels = '0to19 20to39 40to59 60to79 80to99 gt100'.split()

# newdf['LotFrontage'] = pd.cut(newdf['LotFrontage'], bins, labels=labels)

# bins = [1000, 3000, 5000, 7000, 9000, 11000, np.inf]
# labels = '1kto3k 3kto5k 5kto7k 7kto9k 9kto11k gt11k'.split()

# newdf['LotArea'] = pd.cut(newdf['LotArea'], bins, labels=labels)

# bins = [-1, 50, 100, 150, 200, np.inf]
# labels = '0to50 51to100 101to150 151to200 gt200'.split()

# newdf['MasVnrArea'] = pd.cut(newdf['MasVnrArea'], bins, labels=labels)

# bins = [-1, 150, 300, 450, 600, np.inf]
# labels = '0to150 151to300 301to450 451to600 gt600'.split()

# newdf['BsmtFinSF'] = newdf['BsmtFinSF1'] + newdf['BsmtFinSF2']

# newdf['BsmtFinSF'] = pd.cut(newdf['BsmtFinSF'], bins, labels=labels)

# bins = [-1, 200, 400, 600, 800, 1000, np.inf]
# labels = '0to200 201to400 401to600 601to800 801to1000 gt1000'.split()

# newdf['BsmtUnfSF'] = pd.cut(newdf['BsmtUnfSF'], bins, labels=labels)

# bins = [300, 800, 1300, 1800, np.inf]
# labels = '300to800 801to1300 1301to1800 gt1800'.split()

# newdf['1stFlrSF'] = pd.cut(newdf['1stFlrSF'], bins, labels=labels)

# bins = [-1, 300, 600, 900, 1200, np.inf]
# labels = '0to300 301to600 601to900 901to1200 gt1200'.split()

# newdf['2ndFlrSF'] = pd.cut(newdf['2ndFlrSF'], bins, labels=labels)

# bins = [300, 800, 1300, 1800, 2300, np.inf]
# labels = '300to800 801to1300 1301to1800 1801to2300 gt2300'.split()

# newdf['GrLivArea'] = pd.cut(newdf['GrLivArea'], bins, labels=labels)

# bins = [-1, 200, 400, 600, 800, np.inf]
# labels = '0to200 201to400 401to600 601to800 gt800'.split()

# newdf['GarageArea'] = pd.cut(newdf['GarageArea'], bins, labels=labels)

# bins = [-1, 50, 100, 150, 200, 250, np.inf]
# labels = '0to50 51to100 101to150 151to200 201to250 gt250'.split()

# newdf['WoodDeckSF'] = pd.cut(newdf['WoodDeckSF'], bins, labels=labels)
# 

newdf['BsmtFinSF'] = newdf['BsmtFinSF1'] + newdf['BsmtFinSF2']

newdf.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)

colnums = ('LotFrontage LotArea MasVnrArea BsmtFinSF BsmtUnfSF 1stFlrSF 2ndFlrSF GrLivArea GarageArea WoodDeckSF').split()
for col in colnums:
    newdf[col] = pd.cut(newdf[col], 5, ['Div{}'.format(x+1) for x in range(5)])

col_list = ('LotShape LandSlope ExterQual ExterCond BsmtQual BsmtCond BsmtExposure HeatingQC CentralAir '
'KitchenQual FireplaceQu GarageFinish GarageQual GarageCond PavedDrive PoolQC').split()
# 'LotFrontage LotArea MasVnrArea BsmtFinSF BsmtUnfSF 1stFlrSF 2ndFlrSF GrLivArea GarageArea WoodDeckSF').split()

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
            # {'0to19': 0, '20to39': 1, '40to59': 2, '60to79': 3, '80to99': 4, 'gt100': 5},
            # {'1kto3k': 0, '3kto5k': 1, '5kto7k': 2, '7kto9k': 3, '9kto11k': 4, 'gt11k': 5},
            # {'0to50': 0, '51to100': 1, '101to150': 2, '151to200': 3, 'gt200': 4},
            # {'0to150': 0, '151to300': 1, '301to450': 2, '451to600': 3, 'gt600': 4},
            # {'0to200': 0, '201to400': 1, '401to600': 2, '601to800': 3, '801to1000': 4, 'gt1000': 5},
            # {'300to800': 0, '801to1300': 1, '1301to1800': 2, 'gt1800': 3},
            # {'0to300': 0, '301to600': 1, '601to900': 2, '901to1200': 3, 'gt1200': 4},
            # {'300to800': 0, '801to1300': 1, '1301to1800': 2, '1801to2300': 3, 'gt2300': 4},
            # {'0to200': 0, '201to400': 1, '401to600': 2, '601to800': 3, 'gt800': 4},
            # {'0to50': 0, '51to100': 1, '101to150': 2, '151to200': 3, '201to250': 4, 'gt250': 5}]



for i in range(len(col_list)):
    newdf[col_list[i]] = newdf[col_list[i]].map(mapping[i])

newdf = pd.get_dummies(newdf)

trlen = len(train)

train = newdf[:trlen]

dtrain = xgb.DMatrix(train.drop('Id', axis=1), train_label)

test = newdf[trlen:]

label = pd.read_csv('sample_submission.csv')['SalePrice']#.apply(lambda x: math.log10(x))

dtest = xgb.DMatrix(test.drop('Id', axis=1), label)

param = {'max_depth': 3, 'objective': 'reg:squarederror'}
param['nthread'] = 4
param['eval_metric'] = ['rmse', 'mae']

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round=300

model = xgb.train(param, dtrain, num_round, evallist)

print(model.predict(dtest))
sub = pd.DataFrame()
sub['Id'] = test.Id
sub['SalePrice'] = model.predict(dtest)
sub.to_csv('submission_test.csv', index=False)

# cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
#         'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
#         '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
#         'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
#         'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
#         'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'LotShape_num', 'LandSlope_num',
#         'ExterQual_num', 'ExterCond_num', 'BsmtQual_num', 'BsmtCond_num', 'BsmtExposure_num', 
#         'HeatingQC_num', 'CentralAir_num', 'KitchenQual_num', 'FireplaceQu_num',
#         'GarageFinish_num', 'GarageQual_num', 'GarageCond_num', 'PavedDrive_num', 
#         'PoolQC_num', 'MSSubClass_SC120', 'MSSubClass_SC150', 'MSSubClass_SC160', 'MSSubClass_SC180',
#         'MSSubClass_SC190', 'MSSubClass_SC20', 'MSSubClass_SC30', 'MSSubClass_SC40', 'MSSubClass_SC45', 
#         'MSSubClass_SC50', 'MSSubClass_SC60', 'MSSubClass_SC70', 'MSSubClass_SC75',
#         'MSSubClass_SC80', 'MSSubClass_SC85', 'MSSubClass_SC90', 'MSZoning_C (all)', 
#         'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Street_Grvl', 'Street_Pave',
#         'Alley_Grvl', 'Alley_Pave', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 
#         'LandContour_Lvl', 'Utilities_AllPub', 'Utilities_NoSeWa', 'LotConfig_Corner',
#         'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'Neighborhood_Blmngtn',
#         'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide',
#         'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 
#         'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV',
#         'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 
#         'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown',
#         'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 
#         'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker',
#         'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 
#         'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn',
#         'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 
#         'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'BldgType_1Fam',
#         'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 
#         'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf',
#         'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 
#         'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed',
#         'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 
#         'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng',
#         'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 
#         'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd',
#         'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 
#         'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn',
#         'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd',
#         'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other',
#         'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 
#         'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace',
#         'MasVnrType_None', 'MasVnrType_Stone', 'Foundation_BrkTil', 'Foundation_CBlock', 
#         'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtFinType1_ALQ',
#         'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 
#         'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 
#         'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_Floor',
#         'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 
#         'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr',
#         'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 
#         'Functional_Sev', 'Functional_Typ', 'GarageType_2Types', 'GarageType_Attchd',
#         'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 
#         'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'MiscFeature_Gar2', 'MiscFeature_Othr',
#         'MiscFeature_Shed', 'MiscFeature_TenC', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 
#         'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth',
#         'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 
#         'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial']