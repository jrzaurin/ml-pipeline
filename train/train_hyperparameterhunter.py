import numpy as np
import pandas as pd
import warnings
import pickle
import json
import lightgbm as lgb

import pdb

from pathlib import Path
from hyperparameter_hunter import (Environment, CVExperiment,
    BayesianOptimization, Integer, Real, Categorical)
from hyperparameter_hunter import optimization as opt
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


class HHOptimizer(object):
    """docstring for HHOptimizer"""
    def __init__(self, trainDataset, out_dir):

        self.PATH = out_dir
        self.data = trainDataset.data
        self.data['target'] = trainDataset.target
        self.colnames = trainDataset.colnames
        self.categorical_columns = trainDataset.cat_cols + trainDataset.crossed_columns


    def optimize(self, metrics, cv_type, n_splits, maxevals=200, do_predict_proba=None):

        params = self.hyperparameter_space()
        extra_params = self.extra_setup()

        env = Environment(
            train_dataset=self.data,
            results_path=self.PATH,
            metrics=[metrics],
            do_predict_proba = do_predict_proba,
            cv_type=cv_type,
            cv_params=dict(n_splits=n_splits),
        )

        optimizer = opt.BayesianOptimization(iterations=maxevals)
        optimizer.set_experiment_guidelines(
            model_initializer=lgb.LGBMClassifier,
            model_init_params=params,
            model_extra_params=extra_params
        )
        optimizer.go()

        best_experiment = self.PATH+\
            '/HyperparameterHunterAssets/Experiments/Descriptions/'+\
            optimizer.best_experiment+'.json'
        with open(best_experiment) as best:
            best = json.loads(best.read())['hyperparameters']['model_init_params']
        model = lgb.LGBMClassifier(**best)
        X, y = self.data.drop('target',axis=1), self.data.target
        model.fit(X,y,
            feature_name=self.colnames,
            categorical_feature=self.categorical_columns
            )
        pickle.dump(model, open(self.PATH+'/HHmodel.p', 'wb'))
        pickle.dump(optimizer, open(self.PATH+'/HHoptimizer.p', 'wb'))

        return

    def hyperparameter_space(self, param_space=None):

        space = dict(
                is_unbalance = True,
                learning_rate = Real(0.01, 0.3),
                num_boost_round=Categorical(np.arange(50, 500, 20)),
                num_leaves=Categorical(np.arange(31, 256, 4)),
                min_child_weight = Real(0.1, 10),
                colsample_bytree= Real(0.5, 1.),
                subsample=Real(0.5, 1.),
                reg_alpha= Real(0.01, 0.1),
                reg_lambda= Real(0.01, 0.1)
            )

        if param_space:
            return param_space
        else:
            return space


    def extra_setup(self, extra_setup=None):

        extra_params = dict(
            early_stopping_rounds=20,
            feature_name=self.colnames,
            categorical_feature=self.categorical_columns
        )

        if extra_setup:
            return extra_setup
        else:
            return extra_params

# if __name__ == '__main__':

MD_PATH = Path('data/models/')
dtrain = pickle.load(open(MD_PATH/'preprocessor_0_.p', 'rb'))
HHOpt = HHOptimizer(dtrain, str(MD_PATH))
optimizer = HHOpt.optimize('f1_score', StratifiedKFold, n_splits=3, maxevals=3)