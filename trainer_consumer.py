import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer

from utils.messages_utils import publish_traininig_completed
from utils.preprocess_data import build_train
from train.train_hyperopt import LGBOptimizer


KAFKA_HOST = 'localhost:9092'
RETRAIN_TOPIC = 'retrain_topic'
PATH = Path('data/')
TRAIN_DATA = PATH/'train/train.csv'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'
DATAPROCESSOR_ID = 0


def train(model_id, messages):
	print("RETRAINING STARTED (model id: {})".format(model_id))
	dtrain = build_train(TRAIN_DATA, DATAPROCESSORS_PATH, DATAPROCESSOR_ID, messages)
	LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
	LGBOpt.optimize(maxevals=10, model_id=model_id)
	print("RETRAINING COMPLETED (model id: {})".format(model_id))



def start():
	consumer = KafkaConsumer(RETRAIN_TOPIC, bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'retrain' in message and message['retrain']:
			model_id = message['model_id']
			batch_id = message['batch_id']
			message_fname = 'messages_{}_.txt'.format(batch_id)
			messages = MESSAGES_PATH/message_fname
			
			train(model_id, messages)
			publish_traininig_completed(model_id)


if __name__ == '__main__':
	start()