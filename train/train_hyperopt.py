import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import pdb
import warnings

from pathlib import Path
from sklearn.metrics import f1_score
from train.lgb_optimizer import LGBOptimizer

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