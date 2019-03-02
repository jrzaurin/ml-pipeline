import json
import pickle

from kafka import KafkaProducer


def publish_traininig_completed(model_id):
	producer = KafkaProducer(bootstrap_servers='localhost:9092')
	producer.send('retrain_topic', json.dumps({'training_completed': True, 'model_id': model_id}).encode('utf-8'))
	producer.flush()


def read_messages_count(path, repeat_every):
	file_list=list(path.iterdir())
	nfiles = len(file_list)
	if nfiles==0:
		return 0
	else:
		return ((nfiles-1)*repeat_every) + len(file_list[-1].open().readlines())


def append_message(message, path, batch_id):
	message_fname = 'messages_{}_.txt'.format(batch_id)
	f=open(path/message_fname, "a")
	f.write("%s\n" % (json.dumps(message)))
	f.close()


def send_retrain_message(model_id, batch_id):
	producer = KafkaProducer(bootstrap_servers='localhost:9092')
	producer.send('retrain_topic', json.dumps({'retrain': True, 'model_id': model_id, 'batch_id': batch_id}).encode('utf-8'))
	producer.flush()