import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer

from utils import append_message, read_messages_count, send_retrain_message


def reload_model(path):
	return pickle.load(open(path, 'rb'))


PATH = Path('data/')
MD_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'

preprocessor = pickle.load(open(MD_PATH/'train_preprocessor.p', 'rb'))
model = reload_model(MD_PATH/'model.p')

count = read_messages_count(MESSAGES_PATH)

consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
consumer.subscribe(['app_messages', 'retrain_topic'])

for msg in consumer:
	message = json.loads(msg.value)

	if msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']:
		model = reload_model(MD_PATH/'model.p')
		print("NEW MODEL RELOADED")
	elif msg.topic == 'app_messages':
		row = pd.DataFrame(message, index=[0])
		trow = preprocessor.transform(row)
		pred = model.predict(trow)[0]

		append_message(message, MESSAGES_PATH)
		count+=1

		if count % 10 == 0:
			send_retrain_message()

		print(pred)