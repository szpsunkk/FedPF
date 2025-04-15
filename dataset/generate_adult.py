import pandas as pd
from pandas.api.types import CategoricalDtype
import os
import numpy as np
NUM_CLIENT = 20
NUM_CLASS = 2
NUM_SAMPLE_TRAIN = 30000
NUM_SAMPLE_TEST = 3000
DIR = "/home/skk/FL/PFL/PFL-main/dataset/adult/"
# os.makedirs(DIR)
################################## ADULT DATASET ##################################################
df = pd.read_csv('/home/skk/FL/PFL/FNNC/code/Raw_data/adult/adult_train.csv', header= None)
mapping = {' <=50K': 0, ' >50K': 1}
df = df.replace({14: mapping})
df[1] = df[1].replace(' ?', df[1].mode()[0]) # 替换
df[6] = df[6].replace(' ?', df[6].mode()[0])
df[13] = df[13].replace(' ?', df[13].mode()[0])


item = list(df[1].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[1] = df[1].astype(cat_type).cat.codes

item = list(df[6].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[6] = df[6].astype(cat_type).cat.codes

item = list(df[13].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[13] = df[13].astype(cat_type).cat.codes

df[3] = df[3].astype(CategoricalDtype(categories=df[3].unique(), ordered=True)).cat.codes
df[5] = df[5].astype(CategoricalDtype(categories=df[5].unique(), ordered=True)).cat.codes
df[7] = df[7].astype(CategoricalDtype(categories=df[7].unique(), ordered=True)).cat.codes
df[8] = df[8].astype(CategoricalDtype(categories=df[8].unique(), ordered=True)).cat.codes
df[9] = df[9].astype(CategoricalDtype(categories=df[9].unique(), ordered=True)).cat.codes


## Columns 3 and 4 are redundant ####
del df[4]  # 删掉第4列
## noniid dataset
# os.makedirs(DIR + 'noniid/train/')
# for i in range(NUM_CLIENT):
# 	df = df.sample(NUM_SAMPLE_TRAIN, replace=True, random_state=2, axis=0)
# 	file = 'adult_train_noniid' + '_{}_'.format(i) + '.csv'
# 	file_dir = DIR + 'noniid/train/' + file
# 	df.to_csv(file_dir, index=False)

## iid dataset
# os.makedirs(DIR + 'train/')
# os.makedirs(DIR + 'test/')
# train_d = []
# test_d = []

for i in range(NUM_CLIENT):
	train_data = []
	test_data = []
	df = df.sample(NUM_SAMPLE_TRAIN, replace=True, random_state=2, axis=0)
	train_data = {'x': df.to_numpy()[:, :-2], 'y' : df.to_numpy()[:, -1]}
	# train_data = {'x': df.to_numpy()[:30000, :-2], 'y' : df.to_numpy()[:30000, -1]}
	# test_data = {'x': df.to_numpy()[30000:, :-2], 'y' : df.to_numpy()[30000:, -1]}
	# file = '{}'.format(i) + '.npz'
	# file_dir = DIR + 'train/' + file
	with open(DIR + 'train/' + str(i) + '.npz', 'wb') as f:
		np.savez_compressed(f, data=train_data)

	df = df.sample(NUM_SAMPLE_TEST, replace=True, random_state=2, axis=0)
	test_data = {'x': df.to_numpy()[:, :-2], 'y' : df.to_numpy()[:, -1]}
	with open(DIR + 'test/' + str(i) + '.npz', 'wb') as f:
		np.savez_compressed(f, data=test_data)
	# np.savez(file_dir, train_data)
	# df.to_csv(file_dir, index=False)
## iid dataset


# df = pd.read_csv('/home/skk/FL/PFL/FNNC/code/Raw_data/adult/adult.csv', header=None)
# df = df.dropna(axis=1, how='all')
# df.columns = range(15)
# mapping = {'<=50K.': 0, '>50K.': 1}
# df = df.replace({14: mapping})
# df[1] = df[1].replace('?', df[1].mode()[0])
# df[6] = df[6].replace('?', df[6].mode()[0])
# df[13] = df[13].replace('?', df[13].mode()[0])

# df[1] = df[1].astype(CategoricalDtype(categories=list(df[1].unique()), ordered=True)).cat.codes
# df[6] = df[6].astype(CategoricalDtype(categories=list(df[6].unique()), ordered=True)).cat.codes
# df[13] = df[13].astype(CategoricalDtype(categories=list(df[13].unique()), ordered=True)).cat.codes

# df[3] = df[3].astype(CategoricalDtype(categories=df[3].unique(), ordered=True)).cat.codes
# df[5] = df[5].astype(CategoricalDtype(categories=df[5].unique(), ordered=True)).cat.codes
# df[7] = df[7].astype(CategoricalDtype(categories=df[7].unique(), ordered=True)).cat.codes
# df[8] = df[8].astype(CategoricalDtype(categories=df[8].unique(), ordered=True)).cat.codes
# df[9] = df[9].astype(CategoricalDtype(categories=df[9].unique(), ordered=True)).cat.codes
# del df[4]

# # os.makedirs(DIR + 'noniid/test/')
# # for i in range(NUM_CLIENT):
# # 	df = df.sample(NUM_SAMPLE_TEST, replace=True, random_state=2, axis=0)
# # 	file = 'adult_test_noniid' + '_{}_'.format(i) + '.csv'
# # 	file_dir = DIR + 'noniid/test/' + file
# # 	df.to_csv(file_dir, index=False)

# # os.makedirs(DIR + 'test/')
# for i in range(NUM_CLIENT):
# 	test_dataset = []
# 	df = df.sample(NUM_SAMPLE_TEST, replace=False, random_state=2, axis=0)
# 	test_dataset = {'x': df.to_numpy()[:, :-2], 'y' : df.to_numpy()[:, -1]}
# 	with open(DIR + 'test/' + str(i) + '.npz', 'wb') as f:
# 		np.savez_compressed(f, data=test_dataset)
# 	file = '{}'.format(i) + '.npz'
# 	file_dir = DIR + 'test/' + file
# 	np.savez(file_dir, test_dataset)
# 	df.to_csv(file_dir, index=False)