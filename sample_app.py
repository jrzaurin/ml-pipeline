import pandas as pd
import json
import threading
import uuid

from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from time import sleep

# from thread import start_new_thread

PATH = Path('data/')
KAFKA_HOST = 'localhost:9092'
df_test = pd.read_csv(PATH/'adult.test')
# In the real world, the messages would not come with the target/outcome of
# our actions. Here we will keep it and assume that at some point in the
# future we can collect the outcome and monitor how our algorithm is doing
# df_test.drop('income_bracket', axis=1, inplace=True)
df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)
messages = df_test.json.tolist()


def start_producing():
	producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
	for i in range(200):
		message_id = str(uuid.uuid4())
		message = {'request_id': message_id, 'data': json.loads(messages[i])}

		producer.send('app_messages', json.dumps(message).encode('utf-8'))
		producer.flush()

		print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
		sleep(2)


def start_consuming():
	consumer = KafkaConsumer('app_messages', bootstrap_servers=KAFKA_HOST)
	
	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			request_id = message['request_id']
			print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))


threads = []
t = threading.Thread(target=start_producing)
t2 = threading.Thread(target=start_consuming)
threads.append(t)
threads.append(t2)
t.start()
t2.start()
