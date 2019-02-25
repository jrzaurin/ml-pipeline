import pandas as pd
import pickle

from pathlib import Path
from utils import FeatureTools
from sklearn.preprocessing import MinMaxScaler


def build_train(PATH_1, PATH_2=None):
	target = 'income_label'
	df = pd.read_csv(PATH_1)
	if PATH_2:
		df_tmp = pd.read_csv(PATH_2)
		df = pd.concat([df, df_tmp], ignore_index=True)
	df[target] = (df['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
	df.drop('income_bracket', axis=1, inplace=True)

	categorical_columns = list(df.select_dtypes(include=['object']).columns)
	numerical_columns = [c for c in df.columns if c not in categorical_columns+[target]]
	crossed_columns = (['education', 'occupation'], ['native_country', 'occupation'])

	preprocessor = FeatureTools()
	train_preprocessor = preprocessor.fit(df,
		target,
		numerical_columns,
		categorical_columns,
		crossed_columns,
		sc=MinMaxScaler())
	return train_preprocessor


if __name__ == '__main__':

	PATH = Path('data/')
	TR_PATH = PATH/'train'
	MD_PATH = PATH/'models'

	train_preprocessor = build_train(PATH/'adult.data')
	# serialize the preprocessor
	pickle.dump(train_preprocessor, open(MD_PATH/'train_preprocessor.p', 'wb'))

