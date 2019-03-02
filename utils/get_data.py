# -*- coding: utf-8 -*-
# to run: python get_data.py

import pandas as pd
import os

train_path = "data/adult.data"
test_path = "data/adult.test"

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

print("downloading training data...")
df_train = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
    names=COLUMNS, skipinitialspace=True, index_col=0)
df_train.drop("education_num", axis=1, inplace=True)
df_train.to_csv(train_path)

print("downloading testing data...")
df_test = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
    names=COLUMNS, skipinitialspace=True, skiprows=1, index_col=0)
df_test.drop("education_num", axis=1, inplace=True)
df_test.to_csv(test_path)
