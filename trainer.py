import pdb
import json
import pandas as pd
import pickle
import argparse

from pathlib import Path
from kafka import KafkaConsumer

from utils.messages_utils import publish_traininig_completed
from utils.preprocess_data import build_train


KAFKA_HOST = 'localhost:9092'
RETRAIN_TOPIC = 'retrain_topic'
PATH = Path('data/')
TRAIN_DATA = PATH/'train/train.csv'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'


def train(model_id, messages, hyper):
	print("RETRAINING STARTED (model id: {})".format(model_id))
	dtrain = build_train(TRAIN_DATA, DATAPROCESSORS_PATH, model_id, messages)
	if hyper == "hyperopt":
		# from train.train_hyperopt import LGBOptimizer
		from train.train_hyperopt_mlflow import LGBOptimizer
	elif hyper == "hyperparameterhunter":
		# from train.train_hyperparameterhunter import LGBOptimizer
		from train.train_hyperparameterhunter_mlfow import LGBOptimizer
	LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
	LGBOpt.optimize(maxevals=2, model_id=model_id)
	print("RETRAINING COMPLETED (model id: {})".format(model_id))


def start(hyper):
	consumer = KafkaConsumer(RETRAIN_TOPIC, bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'retrain' in message and message['retrain']:
			model_id = message['model_id']
			batch_id = message['batch_id']
			message_fname = 'messages_{}_.txt'.format(batch_id)
			messages = MESSAGES_PATH/message_fname

			train(model_id, messages, hyper)
			publish_traininig_completed(model_id)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--hyper", type=str, default="hyperopt")
	args = parser.parse_args()

	start(args.hyper)