import pandas as pd
import copy


class FeatureTools(object):
	"""Collection of preprocessing methods"""

	@staticmethod
	def num_scaler(df_inp, cols, sc, trained=False):
		"""
		Method to scale numeric columns in a dataframe

		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		cols: List
			List of numeric columns to be scaled
		sc: Scaler object. From sklearn.preprocessing or similar structure
		trained: Boolean
			If True it will only be used to 'transform'

		Returns:
		--------
		df: Pandas.DataFrame
			transformed/normalised dataframe
		sc: trained scaler
		"""
		df = df_inp.copy()
		if not trained:
			df[cols] = sc.fit_transform(df[cols])
		else:
			df[cols] = sc.transform(df[cols])
		return df, sc

	@staticmethod
	def cross_columns(df_inp, x_cols):
		"""
		Method to build crossed columns. These are new columns that are the
		cartesian product of the parent columns.

		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		x_cols: List.
			List of tuples with the columns to cross
			e.g. [('colA', 'colB'),('colC', 'colD')]

		Returns:
		--------
		df: Pandas.DataFrame
			pandas dataframe with the new crossed columns
		colnames: List
			list the new column names
		"""
		df = df_inp.copy()
		colnames = ['_'.join(x_c) for x_c in x_cols]
		crossed_columns = {k:v for k,v in zip(colnames, x_cols)}

		for k, v in crossed_columns.items():
		    df[k] = df[v].apply(lambda x: '-'.join(x), axis=1)

		return df, colnames

	@staticmethod
	def val2idx(df_inp, cols, val_to_idx=None):
		"""
		This is basically a LabelEncoder that returns a dictionary with the
		mapping of the labels.

		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		cols: List
			List of categorical columns to encode
		val_to_idx: Dict
			LabelEncoding dictionary if already exists

		Returns:
		--------
		df: Pandas.DataFrame
			pandas dataframe with the categorical columns encoded
		val_to_idx: Dict
			dictionary with the encoding mappings
		"""
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

	def fit(self, df_inp, target_col, numerical_columns, categorical_columns, x_columns, sc):
		"""
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		target_col: Str
		numerical_columns: List
			List with the numerical columns
		categorical_columns: List
			List with the categorical columns
		x_columns: List
			List of tuples with the columns to cross
		sc: Scaler. From sklearn.preprocessing or object with the same
		structure
		"""
		df = df_inp.copy()
		self.numerical_columns = numerical_columns
		self.categorical_columns = categorical_columns
		self.x_columns = x_columns

		df, self.sc = self.num_scaler(df, numerical_columns, sc)
		df, self.crossed_columns = self.cross_columns(df, x_columns)
		df, self.encoding_d = self.val2idx(df, categorical_columns+self.crossed_columns)

		self.target = df[target_col]
		df.drop(target_col, axis=1, inplace=True)
		self.data = df
		self.colnames = df.columns.tolist()

		return self

	def transform(self, df_inp, trained_sc=None):
		"""
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		trained_sc: Scaler. From sklearn.preprocessing or object with the same

		Returns:
		--------
		df: Pandas.DataFrame
			Tranformed dataframe: scaled, Labelencoded and with crossed columns
		"""
		df = df_inp.copy()
		if trained_sc:
			sc = copy.deepcopy(trained_sc)
		else:
			sc = copy.deepcopy(self.sc)

		df, _ = self.num_scaler(df, self.numerical_columns, sc, trained=True)
		df, _ = self.cross_columns(df, self.x_columns)
		df, _ = self.val2idx(df, self.categorical_columns+self.crossed_columns, self.encoding_d)

		return df
