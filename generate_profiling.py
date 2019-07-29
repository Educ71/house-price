import pandas as pd
import pandas_profiling

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

test = pd.merge(test, pd.read_csv('sample_submission.csv'), on='Id', how='left')

df = pd.concat([train, test], axis=0).reset_index(drop=True)

df.profile_report().to_file('profile_completo.html')