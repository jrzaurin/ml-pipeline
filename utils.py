import pandas as pd
import copy
import json
import pickle

from kafka import KafkaProducer


def publish_traininig_completed():
	producer = KafkaProducer(bootstrap_servers='localhost:9092')
	producer.send('retrain_topic', json.dumps({'training_completed': True}).encode('utf-8'))
	producer.flush()


def read_messages_count(path, repeat_every):
	file_list=list(path.iterdir())
	nfiles = len(file_list)
	if nfiles==0:
		return 0
	else:
		return ((nfiles-1)*repeat_every) + len(file_list[-1].open().readlines())


def append_message(message, path, message_id):
	message_fname = 'messages_{}_.txt'.format(message_id)
	f=open(path/message_fname, "a")
	f.write("%s\n" % (json.dumps(message)))
	f.close()


def send_retrain_message():
	producer = KafkaProducer(bootstrap_servers='localhost:9092')
	producer.send('retrain_topic', json.dumps({'retrain': True}).encode('utf-8'))
	producer.flush()


def reload_model(path):
	return pickle.load(open(path, 'rb'))


class FeatureTools(object):

	@staticmethod
	def num_scaler(df_inp, cols, sc, trained=False):
		df = df_inp.copy()
		if not trained:
			df[cols] = sc.fit_transform(df[cols])
		else:
			df[cols] = sc.transform(df[cols])
		return df, sc

	@staticmethod
	def cross_columns(df_inp, x_cols):
		df = df_inp.copy()
		colnames = ['_'.join(x_c) for x_c in x_cols]
		crossed_columns = {k:v for k,v in zip(colnames, x_cols)}

		for k, v in crossed_columns.items():
		    df[k] = df[v].apply(lambda x: '-'.join(x), axis=1)

		return df, colnames

	@staticmethod
	def val2idx(df_inp, cols, val_to_idx=None):
		df = df_inp.copy()
		if not val_to_idx:

			val_types = dict()
			for c in cols:
			    val_types[c] = df[c].unique()

			val_to_idx = dict()
			for k, v in val_types.items():
			    val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

		for k, v in val_to_idx.items():
		    df[k] = df[k].apply(lambda x: v[x])

		return df, val_to_idx

	def fit(self, df_inp, target_col, num_cols, cat_cols, x_cols, sc):
		df = df_inp.copy()
		self.num_cols = num_cols
		self.cat_cols = cat_cols
		self.x_cols = x_cols

		df, self.sc = self.num_scaler(df, num_cols, sc)
		df, self.crossed_columns = self.cross_columns(df, x_cols)
		df, self.encoding_d = self.val2idx(df, cat_cols+self.crossed_columns)

		self.target = df[target_col]
		df.drop(target_col, axis=1, inplace=True)
		self.data = df
		self.colnames = df.columns.tolist()

		return self

	def transform(self, df_inp, trained_sc=None):
		df = df_inp.copy()
		if trained_sc:
			sc = copy.deepcopy(trained_sc)
		else:
			sc = copy.deepcopy(self.sc)

		df, _ = self.num_scaler(df, self.num_cols, sc, trained=True)
		df, _ = self.cross_columns(df, self.x_cols)
		df, _ = self.val2idx(df, self.cat_cols+self.crossed_columns, self.encoding_d)

		return df

