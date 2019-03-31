import pdb
import json
import pandas as pd
import pickle

import boto3
import botocore

from pathlib import Path
from kafka import KafkaConsumer
from utils.messages_utils import append_message, read_messages_count, send_retrain_message, publish_prediction

KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages', 'retrain_topic']
PATH = Path('data/')
MODELS_PATH = PATH/'models'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MESSAGES_PATH = PATH/'messages'
RETRAIN_EVERY = 25
EXTRA_MODELS_TO_KEEP = 1

BUCKET_NAME='ml_pipeline'

column_order = load_column_order_s3()
# dataprocessor = None
consumer = None
model = None


# def reload_model(path):
# 	return pickle.load(open(path, 'rb'))


def is_retraining_message(msg):
	message = json.loads(msg.value)
	return msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']


def is_application_message(msg):
	message = json.loads(msg.value)
	return msg.topic == 'app_messages' and 'prediction' not in message


def load_column_order_s3():
	try:
		s3 = boto3.resource('s3')

		Key = "column_order.p"
		outPutName = "column_order.p"

	    s3.Bucket(BUCKET_NAME).download_file(Key, outPutName)
	except botocore.exceptions.ClientError as e:
	    if e.response['Error']['Code'] == "404":
	        print("The object does not exist.")
	    else:
	        raise


# def load_dataprocessor_s3(dataprocessor_id):
# 	dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
# 	Key = 'processors/{}'.format(dataprocessor_fname)
# 	outPutName = DATAPROCESSORS_PATH/dataprocessor_fname

# 	s3 = boto3.resource('s3')
# 	try:
# 	    s3.Bucket(BUCKET_NAME).download_file(Key, outPutName)
# 	except botocore.exceptions.ClientError as e:
# 	    if e.response['Error']['Code'] == "404":
# 	        print("The dataprocessor object {} does not exist.".format(Key))
# 	    else:
# 	        raise


def predict(message, column_order):
	row = pd.DataFrame(message, index=[0])
	# sanity check
	assert row.columns.tolist()[:-1] == column_order
	# In the real world we would not have the target (here 'income_bracket').
	# In this example we keep it and we will retrain the model as it reads
	# RETRAIN_EVERY number of messages. In the real world, after RETRAIN_EVERY
	# number of messages have been collected, one would have to wait until we
	# can collect RETRAIN_EVERY targets AND THEN retrain
	row.drop('income_bracket', axis=1, inplace=True)
	
	# trow = dataprocessor.transform(row)
	# return model.predict(trow)[0]

	response = sagemaker_client.invoke_endpoint(EndpointName='lightgbm-ml_pipeline',
	                                  			Body=json.dumps(row.to_json()),
	                                  			ContentType='application/json')
	return response['Body'].read()


def start(model_id, messages_count, batch_id, sagemaker_client):
	for msg in consumer:
		message = json.loads(msg.value)

		if is_retraining_message(msg):
		# 	model_fname = 'model_{}_.p'.format(model_id)
		# 	model = reload_model(MODELS_PATH/model_fname)
			print("NEW MODEL RELOADED {}".format(model_id))

		elif is_application_message(msg):
			request_id = message['request_id']
			pred = predict(message['data'], column_order, sagemaker_client)
			publish_prediction(pred, request_id)

			append_message(message['data'], MESSAGES_PATH, batch_id)
			messages_count += 1
			if messages_count % RETRAIN_EVERY == 0:
				model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
				send_retrain_message(model_id, batch_id)
				batch_id += 1


if __name__ == '__main__':
	# dataprocessor_id = 0
	# dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
	# dataprocessor = pickle.load(open(DATAPROCESSORS_PATH/dataprocessor_fname, 'rb'))

	messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
	batch_id = messages_count % RETRAIN_EVERY

	model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
	model_fname = 'model_{}_.p'.format(model_id)
	model = reload_model(MODELS_PATH/model_fname)

	consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
	consumer.subscribe(TOPICS)

	sagemaker_client = boto3.client('runtime.sagemaker')

	start(model_id, messages_count, batch_id, sagemaker_client)
