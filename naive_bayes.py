import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, tree


COLUMNS = ["age", "work class", "education", "education-num", "marital_status", "occ_code", "relationship", "race", "sex", "cap_gain", "cap_loss", "hours_per_week", "native_country"]

#reading training and test data
TRAININGSET = pd.read_csv('income.training.csv')
TESTSET = pd.read_csv('income-test.csv')

#filling Nan's with -999999
fillNans_training_data.fillna(-99999)
fillNans_in_te.fillna(-99999)
