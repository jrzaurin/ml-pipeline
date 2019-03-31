import pdb
import json
import pandas as pd
import pickle
import argparse

import sagemaker as sage
from time import gmtime, strftime

from pathlib import Path
from kafka import KafkaConsumer

from utils.messages_utils import publish_traininig_completed
from utils.preprocess_data import build_train


KAFKA_HOST = 'localhost:9092'
RETRAIN_TOPIC = 'retrain_topic'
PATH = Path('data/')
TRAIN_DATA = PATH/'train/train.csv'
# DATAPROCESSORS_PATH = PATH/'dataprocessors'
# MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'


def train(model_id, messages, hyper):
	print("RETRAINING STARTED (model id: {})".format(model_id))
	# dtrain = build_train(TRAIN_DATA, DATAPROCESSORS_PATH, model_id, messages)
	# if hyper == "hyperopt":
	# 	# from train.train_hyperopt import LGBOptimizer
	# 	from train.train_hyperopt_mlflow import LGBOptimizer
	# elif hyper == "hyperparameterhunter":
	# 	# from train.train_hyperparameterhunter import LGBOptimizer
	# 	from train.train_hyperparameterhunter_mlfow import LGBOptimizer
	# LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
	# LGBOpt.optimize(maxevals=2, model_id=model_id)
	
	sess = sage.Session()
	
	# 1 - Upload data to S3
	train_data_location = sess.upload_data(messages, key_prefix=prefix)
	
	# 2 - Call SageMaker algorithm
	account = sess.boto_session.client('sts').get_caller_identity()['Account']
	region = sess.boto_session.region_name
	image = '{}.dkr.ecr.{}.amazonaws.com/ml-pipeline-lightgbm-hyperopt:latest'.format(account, region)

	lightgbm_sagemaker = sage.estimator.Estimator(image,
	                       role, 1, 'ml.c4.2xlarge',
	                       output_path="s3://{}/output".format(sess.default_bucket()),
	                       sagemaker_session=sess,
	                       hyperparameters={'model_id': model_id})
	
	lightgbm_sagemaker.fit(train_data_location)

	# Create endpoint config
	session = lightgbm_sagemaker.sagemaker_session

	container_def = lightgbm_sagemaker.prepare_container_def(instance_type='ml.m4.xlarge')
	model_name = str(random.random())[2:]
	session.create_model(model_name, role, container_def)

	config_name = str(random.random())[2:]
	session.create_endpoint_config(name=config_name,
	                              model_name=model_name,
	                              initial_instance_count=1,
	                              instance_type='ml.m4.xlarge')
	
	# Update desired endpoint with new Endpoint Config
	client = boto3.client('sagemaker')
	client.update_endpoint(EndpointName='lightgbm-ml_pipeline',
	                       EndpointConfigName=config_name)

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