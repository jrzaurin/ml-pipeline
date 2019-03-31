import pandas as pd
import json

from pathlib import Path
from kafka import KafkaProducer
from time import sleep

PATH = Path('data/')
df_test = pd.read_csv(PATH/'adult.test')
# In the real world, the messages would not come with the target/outcome of
# our actions. Here we will keep it and assume that at some point in the
# future we can collect the outcome and monitor how our algorithm is doing
# df_test.drop('income_bracket', axis=1, inplace=True)
df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)
messages = df_test.json.tolist()
print(messages[0])

# producer = KafkaProducer(bootstrap_servers='localhost:9092')
# for i in range(200):
# 	producer.send('app_messages', messages[i].encode('utf-8'))
# 	producer.flush()
# 	sleep(2)