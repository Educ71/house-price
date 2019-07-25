import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_profiling
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')

train_label = train['SalePrice']#.apply(lambda x: math.log10(x))

train.drop('SalePrice', axis=1, inplace=True)

test = pd.read_csv('test.csv')

newdf = pd.concat([train, test], axis=0).reset_index(drop=True)

newdf['MSSubClass'] = newdf['MSSubClass'].apply(lambda x: 'SC{}'.format(x))
newdf['GarageYrBlt'] = newdf['GarageYrBlt'].apply(lambda x: 2007 if x == 2207 else x)
colnums = newdf.select_dtypes(include=[np.number]).columns.values
newdf[colnums] = newdf[colnums].fillna(0)
colcats = newdf.select_dtypes(include=[np.object]).columns.values
newdf[colcats] = newdf[colcats].fillna('None')

# newdf.profile_report().to_file('profile_completo_bef.html')

newdf['HasAlley'] = newdf['Alley'].apply(lambda x: 0 if x == 'None' else 1)

newdf['BsmtFinSF'] = newdf['BsmtFinSF1'] + newdf['BsmtFinSF2']

newdf['HasPorch'] = newdf['OpenPorchSF'] + newdf['EnclosedPorch'] + newdf['3SsnPorch'] + newdf['ScreenPorch']

newdf['HasPorch'] = newdf['HasPorch'].apply(lambda x: 1 if x > 0 else 0)

newdf['HasMasVnr'] = newdf['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)

newdf['HasFireplace'] = newdf['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

newdf['Has2ndFlr'] = newdf['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

mapping = [{'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
            {'Gtl': 2, 'Mod': 1, 'Sev': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
            {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            # {'N': 0, 'Y': 1},
            {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
            {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
            {'Y': 2, 'P': 1, 'N': 0}]

bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, np.inf]
labels = ['{}to{}'.format(x, x+5) for x in range(0, 100, 5)] + ['gt100']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['LotFrontage'] = pd.cut(newdf['LotFrontage'], bins, labels=labels)

bins = [x for x in range(1000, 11001, 500)] + [np.inf]
labels = ['{}kto{}k'.format(x/10, (x+5)/10) for x in range(10, 110, 5)] + ['gt11k']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['LotArea'] = pd.cut(newdf['LotArea'], bins, labels=labels)

bins = [-1] + [x for x in range(20, 650, 20)] + [np.inf]
labels = ['{}to{}'.format(x, x+20) for x in range(0, 640, 20)] + ['gt640']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['BsmtFinSF'] = newdf['BsmtFinSF1'] + newdf['BsmtFinSF2']

newdf['BsmtFinSF'] = pd.cut(newdf['BsmtFinSF'], bins, labels=labels)

bins = [-1] + [x for x in range(40, 1000, 40)] + [np.inf]
labels = ['{}to{}'.format(x, x+40) for x in range(40, 1000, 40)] + ['gt1000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['BsmtUnfSF'] = pd.cut(newdf['BsmtUnfSF'], bins, labels=labels)

bins = [-1] + [x for x in range(50, 1250, 50)] + [np.inf]
labels = ['{}to{}'.format(x, x+40) for x in range(50, 1250, 50)] + ['gt1000']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['TotalBsmtSF'] = pd.cut(newdf['TotalBsmtSF'], bins, labels=labels)

bins = [x for x in range(300, 1801, 50)] + [np.inf]
labels = ["{}to{}".format(x, x+50) for x in range(300, 1800, 50)] + ['gt1800']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['1stFlrSF'] = pd.cut(newdf['1stFlrSF'], bins, labels=labels)

# bins = [-1] + [x for x in range(50, 1500, 50)] + [np.inf]
# labels = ["{}to{}".format(x, x+50) for x in range(0, 1420, 50)] + ['gt1450']

# labs = {labels[value]: value for value in range(len(labels))}
# mapping.append(labs)

# newdf['2ndFlrSF'] = pd.cut(newdf['2ndFlrSF'], bins, labels=labels)

bins = [x for x in range(300, 2301, 75)] + [np.inf]
labels = ["{}to{}".format(x, x+75) for x in range(300, 2250, 75)] + ['gt2250']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['GrLivArea'] = pd.cut(newdf['GrLivArea'], bins, labels=labels)

bins = [-1] + [x for x in range(25, 801, 25)] + [np.inf]
labels = ["{}to{}".format(x, x+25) for x in range(0, 800, 25)] + ['gt800']

labs = {labels[value]: value for value in range(len(labels))}
mapping.append(labs)

newdf['GarageArea'] = pd.cut(newdf['GarageArea'], bins, labels=labels)

# bins = [-1] + [x for x in range(20, 250, 20)] + [np.inf]
# labels = '0to50 51to100 101to150 151to200 201to250 gt250'.split()

# newdf['WoodDeckSF'] = pd.cut(newdf['WoodDeckSF'], bins, labels=labels)

newdf['HasWoodDeck'] = newdf['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)

newdf.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType2'], axis=1, inplace=True)

# colnums = ('LotFrontage LotArea MasVnrArea BsmtFinSF BsmtUnfSF 1stFlrSF 2ndFlrSF GrLivArea GarageArea WoodDeckSF').split()
# for col in colnums:
#     newdf[col] = pd.cut(newdf[col], 5, ['Div{}'.format(x+1) for x in range(5)])

col_list = ('LotShape LandSlope ExterQual ExterCond BsmtQual BsmtCond BsmtExposure HeatingQC '#CentralAir '
            'KitchenQual FireplaceQu GarageFinish GarageCond PavedDrive' #).split()
            ' LotFrontage LotArea BsmtFinSF BsmtUnfSF TotalBsmtSF 1stFlrSF GrLivArea GarageArea').split()

for i in range(len(col_list)):
    newdf[col_list[i]] = newdf[col_list[i]].map(mapping[i])

newdf.drop(['GarageQual', 'Street', 'GarageYrBlt', 'Alley', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolQC', 'RoofMatl', 'Utilities', 'PoolArea', 'LowQualFinSF', 'MasVnrArea', 'MasVnrType', 'Condition2', 'Heating', 'MiscFeature', 'MiscVal', 'Fireplaces', 'FireplaceQu', 'HasAlley', 'Functional', 'BsmtHalfBath', 'WoodDeckSF', '2ndFlrSF', 'CentralAir'], axis=1, inplace=True)
newdf.profile_report().to_file('profile_completo_af.html')
newdf = pd.get_dummies(newdf)
newdf.columns = [x.replace(" ", "_") for x in newdf.columns.values]

trlen = len(train)

train = newdf[:trlen]

X_train, X_test, y_train, y_test = train_test_split(train, train_label, test_size=0.3, random_state=50)

dtrain = xgb.DMatrix(X_train.drop('Id', axis=1), y_train)
dtest = xgb.DMatrix(X_test.drop('Id', axis=1), y_test)

def rmsle(pred: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    pred[pred < -1] = -1 + 1e-6
    error = float(np.sqrt(np.sum(np.power(np.log1p(y) - np.log1p(pred), 2)) / len(y)))
    return 'MyRMSLE', error

param = {'max_depth': 2, 'objective': 'reg:squarederror', 'disable_default_eval_metric': 1}
param['nthread'] = 4
# param['eval_metric'] = "rmsle"

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round=300

# xgb_model = xgb.XGBRegressor()
# xgb.cv(param, dtrain, num_round, nfold=5, feval=rmsle, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
xgb_model = xgb.train(param, dtrain, num_round, evallist, feval=rmsle)

test = newdf[trlen:]

label = pd.read_csv('sample_submission.csv')['SalePrice']#.apply(lambda x: math.log10(x))

dsub = xgb.DMatrix(test.drop('Id', axis=1), label)

print(xgb_model.predict(dsub))
sub = pd.DataFrame()
sub['Id'] = test.Id
sub['SalePrice'] = xgb_model.predict(dsub)
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