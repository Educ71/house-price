import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# train = pd.read_csv('train.csv')

# train = train[train.GrLivArea < 4500]
# train = train[train.LotArea < 10000]

# test = pd.read_csv('test.csv')

# test = pd.merge(test, pd.read_csv('sample_submission.csv'), on='Id', how='left')

# newdf = pd.concat([train, test], axis=0).reset_index(drop=True)

# newdf.profile_report().to_file('profile_completo_bef.html')

# plt.scatter(train.GrLivArea, train.SalePrice)
# plt.show()

# plt.scatter(train.LotArea, train.SalePrice)
# plt.show()

df = pd.read_csv('teste_file.csv')

df.profile_report().to_file('profile_completo_af.html')
