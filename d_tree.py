import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, tree


COLUMNS = ["age", "work class", "education", "education-num", "marital_status", "occ_code", "relationship", "race", "sex", "cap_gain", "cap_loss", "hours_per_week", "native_country"]

#reading training and test data
df_train = pd.read_csv('income-training.txt', names = COLUMNS)
df_test = pd.read_csv('income-test.txt', names = COLUMNS, skiprows =1)

#filling Nan's with -999999
df_train.fillna(-99999)
df_test.fillna(-99999)

le = preprocessing.LabelEncoder()

for x in Columns:
    if df_train[x].dtypes == 'object'
        data = df_train[x].append(df_test[x])
        le.fit(data.values)
        df_train[x] = le.transform(df_train[x])
        df_test[x] = le.transform(df_test[x])


X = [[25, Private, 11th, 7, Never-married, Machine-op-inspect, Own-child, Black, Male, 0, 0, 40, United-States, <=50k]]
