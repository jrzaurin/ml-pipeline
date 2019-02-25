import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer

from utils import publish_traininig_completed, load_training_data
from train import LGBOptimize


PATH = Path('data/')
MD_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'

preprocessor = pickle.load(open(MD_PATH/'train_preprocessor.p', 'rb'))
model = pickle.load(open(MD_PATH/'model.p', 'rb'))

consumer = KafkaConsumer('retrain_topic', bootstrap_servers='localhost:9092')
for msg in consumer:
	message = json.loads(msg.value)
	if 'retrain' in message and message['retrain']:
		print("RETRAINING")
		dtrain = load_training_data(MESSAGES_PATH)
		LGBOpt = LGBOptimize(dtrain, MD_PATH)
		LGBOpt.optimize(maxevals=200)
		print("RETRAINING COMPLETED")
		publish_traininig_completed()