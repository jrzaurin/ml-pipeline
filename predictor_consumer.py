import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer
from utils import append_message, read_messages_count, send_retrain_message, reload_model


PATH = Path('data/')
MODELS_PATH = PATH/'models'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MESSAGES_PATH = PATH/'messages'
retrain_every = 50

model_id, dataprocessor_id, message_id = [0]*3
model_fname = 'model_{}_.p'.format(model_id)
dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
message_fname = 'messages_{}_.txt'.format(message_id)

dataprocessor = pickle.load(open(DATAPROCESSORS_PATH/dataprocessor_fname, 'rb'))
model = reload_model(MODELS_PATH/model_fname)
count = read_messages_count(MESSAGES_PATH, retrain_every)

consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
consumer.subscribe(['app_messages', 'retrain_topic'])
for msg in consumer:
	message = json.loads(msg.value)
	if msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']:
		model = reload_model(MODELS_PATH/model_fname)
		print("NEW MODEL RELOADED")
		model_id+=1
		model_fname = 'model_{}_.p'.format(model_id)
	elif msg.topic == 'app_messages':
		row = pd.DataFrame(message, index=[0])
		row.drop('income_bracket', axis=1, inplace=True)
		trow = dataprocessor.transform(row)
		pred = model.predict(trow)[0]
		append_message(message, MESSAGES_PATH, message_id)
		count+=1
		if count % retrain_every == 0:
			send_retrain_message()
			message_id+=1
		print('observation number: {}. Prediction: {}'.format(count,pred))