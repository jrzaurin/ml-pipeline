import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import warnings
import argparse
import os
import pdb

from pathlib import Path
from utils.preprocess_data import build_train


PATH = Path('data/')
TRAIN_PATH = PATH/'train'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'


def create_folders():
	print("creating directory structure...")
	(PATH).mkdir(exist_ok=True)
	(TRAIN_PATH).mkdir(exist_ok=True)
	(MODELS_PATH).mkdir(exist_ok=True)
	(DATAPROCESSORS_PATH).mkdir(exist_ok=True)
	(MESSAGES_PATH).mkdir(exist_ok=True)


def download_data():
	train_path = PATH/'adult.data'
	test_path = PATH/'adult.test'

	COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
	           "marital_status", "occupation", "relationship", "race", "gender",
	           "capital_gain", "capital_loss", "hours_per_week", "native_country",
	           "income_bracket"]

	print("downloading training data...")
	df_train = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
	    names=COLUMNS, skipinitialspace=True, index_col=0)
	df_train.drop("education_num", axis=1, inplace=True)
	df_train.to_csv(train_path)
	df_train.to_csv(PATH/'train/train.csv')

	print("downloading testing data...")
	df_test = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
	    names=COLUMNS, skipinitialspace=True, skiprows=1, index_col=0)
	df_test.drop("education_num", axis=1, inplace=True)
	df_test.to_csv(test_path)


def create_data_processor():
	print("creating preprocessor...")
	dataprocessor = build_train(TRAIN_PATH/'train.csv', DATAPROCESSORS_PATH)


def create_model(hyper):
	print("creating model...")
	init_dataprocessor = 'dataprocessor_0_.p'
	dtrain = pickle.load(open(DATAPROCESSORS_PATH/init_dataprocessor, 'rb'))
	if hyper == "hyperopt":
		# from train.train_hyperopt import LGBOptimizer
		from train.train_hyperopt_mlflow import LGBOptimizer
	elif hyper == "hyperparameterhunter":
		# from train.train_hyperparameterhunter import LGBOptimizer
		from train.train_hyperparameterhunter_mlfow import LGBOptimizer
	LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
	LGBOpt.optimize(maxevals=2)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--hyper", type=str, default="hyperopt")
	args = parser.parse_args()
	create_folders()
	download_data()
	create_data_processor()
	create_model(args.hyper)