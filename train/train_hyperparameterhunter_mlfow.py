import numpy as np
import pandas as pd
import warnings
import pickle
import json
import lightgbm as lgb
import mlflow
import mlflow.sklearn

import pdb

from pathlib import Path
from hyperparameter_hunter import (Environment, CVExperiment,
    BayesianOptimization, Integer, Real, Categorical)
from hyperparameter_hunter import optimization as opt
from sklearn.model_selection import StratifiedKFold
from mlflow.tracking import MlflowClient


warnings.filterwarnings("ignore")


class LGBOptimizer(object):
    def __init__(self, trainDataset, out_dir):
        """
        Hyper Parameter optimization

        Comments: Hyperparameter_hunter (hereafter HH) is a fantastic package
        (https://github.com/HunterMcGushion/hyperparameter_hunter) to avoid
        wasting time as you optimise parameters. In the words of his author:
        "For so long, hyperparameter optimization has been such a time
        consuming process that just pointed you in a direction for further
        optimization, then you basically had to start over".

        Parameters:
        -----------
        trainDataset: FeatureTools object
            The result of running FeatureTools().fit()
        out_dir: Str
            Path to the output directory
        """

        self.PATH = out_dir
        self.data = trainDataset.data
        self.data['target'] = trainDataset.target
        self.colnames = trainDataset.colnames
        self.categorical_columns = trainDataset.categorical_columns + trainDataset.crossed_columns

    def optimize(self, metrics, cv_type, n_splits, maxevals=200, do_predict_proba=None):

        params = self.hyperparameter_space()
        extra_params = self.extra_setup()

        env = Environment(
            train_dataset=self.data,
            results_path='HyperparameterHunterAssets',
            # results_path=self.PATH,
            metrics=[metrics],
            do_predict_proba = do_predict_proba,
            cv_type=cv_type,
            cv_params=dict(n_splits=n_splits),
        )

        # optimizer = opt.GradientBoostedRegressionTreeOptimization(iterations=maxevals)
        optimizer = opt.BayesianOptimization(iterations=maxevals)
        optimizer.set_experiment_guidelines(
            model_initializer=lgb.LGBMClassifier,
            model_init_params=params,
            model_extra_params=extra_params
        )
        optimizer.go()

        # there are a few fixes on its way and the next few lines will soon be
        # one. At the moment, to access to the best parameters one has to read
        # from disc and access them
        best_experiment = 'HyperparameterHunterAssets/Experiments/Descriptions/'+\
            optimizer.best_experiment+'.json'
        with open(best_experiment) as best:
            best = json.loads(best.read())['hyperparameters']['model_init_params']

        # The next few lines are the only ones related to mlflow. One
        # "annoying" behaviour of mlflow is that when you instantiate a client
        # it creates the 'mlruns' dir by default as well as the first
        # experiment and there does not seem to be a way I can change this
        # behaviour without changing the source code. The solution is the
        # following hack:
        if not Path('mlruns').exists():
            client = MlflowClient()
        else:
            client = MlflowClient()
            n_experiments = len(client.list_experiments())
            client.create_experiment(name=str(n_experiments))
        experiments = client.list_experiments()
        with mlflow.start_run(experiment_id=experiments[-1].experiment_id) as run:
            model = lgb.LGBMClassifier(**best)
            X, y = self.data.drop('target',axis=1), self.data.target
            model.fit(X,y,
                feature_name=self.colnames,
                categorical_feature=self.categorical_columns
                )
            for name, value in best.items():
                mlflow.log_param(name, value)
            mlflow.log_metric('f1_score', -optimizer.optimizer_result.fun)
            mlflow.sklearn.log_model(model, "model")

        pickle.dump(model, open(self.PATH+'/HHmodel.p', 'wb'))
        pickle.dump(optimizer, open(self.PATH+'/HHoptimizer.p', 'wb'))

        return

    def hyperparameter_space(self, param_space=None):

        space = dict(
                is_unbalance = True,
                learning_rate = Real(0.01, 0.3),
                num_boost_round=Integer(50, 500),
                num_leaves=Integer(31, 255),
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

#     MD_PATH = Path('data/models/')
#     dtrain = pickle.load(open(MD_PATH/'preprocessor_0_.p', 'rb'))
#     HHOpt = HHOptimizer(dtrain, str(MD_PATH))
#     optimizer = HHOpt.optimize('f1_score', StratifiedKFold, n_splits=3, maxevals=3)