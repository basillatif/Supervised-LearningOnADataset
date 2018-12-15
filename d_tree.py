import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, tree
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, Normalizer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

COLUMNS = ["age", "work class", "education", "education-num", "marital_status", "occ_code", "relationship", "race", "sex", "cap_gain", "cap_loss", "hours_per_week", "native_country", "income"]  # noqa: E501

# reading training and test data
df_train = pd.read_csv('income-training.csv', names=COLUMNS)
df_test = pd.read_csv('income-test.csv', names=COLUMNS)

# f illing Nan's with -999999
df_train.fillna(-99999)
df_test.fillna(-99999)

le = preprocessing.LabelEncoder()
ohe = OneHotEncoder(sparse=False)
ordinal = OrdinalEncoder()
encoder = le

# fit and transform
for x in COLUMNS:
    if df_train[x].dtypes == 'object':
        df_train_2d = df_train[[x]].copy()
        df_test_2d = df_test[[x]].copy()
        df_train[x] = encoder.fit_transform(df_train_2d)
        df_test[x] = encoder.fit_transform(df_test_2d)

print(df_train)

# ct = ColumnTransformer([("norm1", KBinsDiscretizer(n_bins=5, encode=’ordinal’, strategy=’quantile’), [0, 1]), ("norm2", Normalizer(norm='l1'), slice(2, 4))])


# clf = DecisionTreeClassifier(random_state=0)
# score = cross_val_score(clf, df_train, COLUMNS)
# X = [[25, Private, 11th, 7, Never-married, Machine-op-inspect, Own-child, Black, Male, 0, 0, 40, United-States, <=50k]]
