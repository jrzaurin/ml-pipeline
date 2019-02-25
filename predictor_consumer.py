import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer

PATH = Path('data/')
MD_PATH = PATH/'models'

preprocessor = pickle.load(open(MD_PATH/'train_preprocessor.p', 'rb'))
model = pickle.load(open(MD_PATH/'model.p', 'rb'))

consumer = KafkaConsumer('my_favorite_topic', bootstrap_servers='localhost:9092')
for msg in consumer:
	row = pd.DataFrame(json.loads(msg.value), index=[0])
	trow = preprocessor.transform(row)
	pred = model.predict(trow)[0]
	print(pred)

