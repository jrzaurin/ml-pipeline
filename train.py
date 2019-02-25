import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import pdb
import warnings

from pathlib import Path
from hyperopt import hp, tpe, fmin, Trials
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


def best_threshold(y_true, pred_proba, proba_range, verbose=False):
	scores = []
	for prob in proba_range:
		pred = [int(p>prob) for p in pred_proba]
		score = f1_score(y_true,pred)
		scores.append(score)
		if verbose:
			print("INFO: prob threshold: {}.  score :{}".format(round(prob,3), round(score,5)))
	best_score = scores[np.argmax(scores)]
	optimal_threshold = proba_range[np.argmax(scores)]
	return (optimal_threshold, best_score)


def lgb_f1_score(preds, lgbDataset):
	binary_preds = [int(p>0.5) for p in preds]
	y_true = lgbDataset.get_label()
	# lightgbm: (eval_name, eval_result, is_higher_better)
	return 'f1', f1_score(y_true, binary_preds), True


class LGBOptimize(object):
	"""
	Hyper Parameter optimization
	"""
	def __init__(self, trainDataset, out_dir):

		self.PATH = out_dir
		self.early_stop_dict = {}

		self.X = trainDataset.data
		self.y = trainDataset.target
		self.colnames = trainDataset.colnames
		self.categorical_columns = trainDataset.cat_cols + trainDataset.crossed_columns

		self.lgtrain = lgb.Dataset(self.X,label=self.y,
			feature_name=self.colnames,
			categorical_feature = self.categorical_columns,
			free_raw_data=False)

	def optimize(self, maxevals=50):

		param_space = self.hyperparameter_space()
		objective = self.get_objective(self.lgtrain)
		objective.i=0
		trials = Trials()
		best = fmin(fn=objective,
		            space=param_space,
		            algo=tpe.suggest,
		            max_evals=maxevals,
		            trials=trials)
		best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
		best['num_leaves'] = int(best['num_leaves'])
		best['verbose'] = -1

		model = lgb.LGBMClassifier(**best)
		model.fit(self.lgtrain.data,
			self.lgtrain.label,
			feature_name=self.colnames,
			categorical_feature=self.categorical_columns)

		pickle.dump(model, open(self.PATH/'model.p', 'wb'))
		pickle.dump(best, open(self.PATH/'optimal_parameters.p', 'wb'))

		self.best = best
		self.model = model


	def get_objective(self, train):

		def objective(params):
			"""
			objective function for lightgbm.
			"""
			# hyperopt casts as float
			params['num_boost_round'] = int(params['num_boost_round'])
			params['num_leaves'] = int(params['num_leaves'])

			# need to be passed as parameter
			params['is_unbalance'] = True
			params['verbose'] = -1
			params['seed'] = 1

			cv_result = lgb.cv(
				params,
				train,
				num_boost_round=params['num_boost_round'],
				metrics='binary_logloss',
				# feval = lgb_f1_score,
				nfold=3,
				stratified=True,
				early_stopping_rounds=20)
			self.early_stop_dict[objective.i] = len(cv_result['binary_logloss-mean'])
			error = round(cv_result['binary_logloss-mean'][-1], 4)
			objective.i+=1
			return error

		return objective

	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
			'num_boost_round': hp.quniform('num_boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
		    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
		}

		if param_space:
			return param_space
		else:
			return space

if __name__ == '__main__':

	MD_PATH = Path('data/models/')
	dtrain = pickle.load(open(MD_PATH/'train_preprocessor.p', 'rb'))
	LGBOpt = LGBOptimize(dtrain, MD_PATH)
	LGBOpt.optimize(maxevals=200)