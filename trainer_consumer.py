import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer

from utils import publish_traininig_completed
from preprocess_data import build_train
from train_hyperopt import LGBOptimizer

PATH = Path('data/')
TRAIN_DATA = PATH/'train/train.csv'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'

model_id, dataprocessor_id, message_id = [0]*3
model_fname = 'model_{}_.p'.format(model_id)
message_fname = 'messages_{}_.txt'.format(message_id)

model = pickle.load(open(MODELS_PATH/model_fname, 'rb'))
model_id+=1
messages = MESSAGES_PATH/message_fname

consumer = KafkaConsumer('retrain_topic', bootstrap_servers='localhost:9092')
for msg in consumer:
	message = json.loads(msg.value)
	if 'retrain' in message and message['retrain']:
		print("RETRAINING")
		dtrain = build_train(TRAIN_DATA, DATAPROCESSORS_PATH, model_id, messages)
		LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
		LGBOpt.optimize(maxevals=10, model_id=model_id)
		print("RETRAINING COMPLETED")
		publish_traininig_completed()
		model_id+=1
		message_id+=1
		message_fname = 'messages_{}_.txt'.format(message_id)
		messages = MESSAGES_PATH/message_fname
