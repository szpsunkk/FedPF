import pandas as pd
from pandas.api.types import CategoricalDtype
import os
import numpy as np
NUM_CLIENT = 20
NUM_CLASS = 2
NUM_SAMPLE_TRAIN = 30000
NUM_SAMPLE_TEST = 3000
DIR = "/home/skk/FL/PFL/PFL-main/dataset/bank/"

df = pd.read_csv('/home/skk/FL/PFL/PFL-main/dataset/bank/UCI_Credit_Card.csv', header= None)
print(df)
del df[0]
# del df[1]


df = df.iloc[1: , :]
df = df.astype('float32')

for i in range(NUM_CLIENT):
	train_data = []
	test_data = []
	df = df.sample(NUM_SAMPLE_TRAIN, replace=True, random_state=2, axis=0)
	train_data = {'x': df.to_numpy()[:, :-2], 'y' : df.to_numpy()[:, -1]}
	with open(DIR + 'train/' + str(i) + '.npz', 'wb') as f:
		np.savez_compressed(f, data=train_data)

	df = df.sample(NUM_SAMPLE_TEST, replace=True, random_state=2, axis=0)
	test_data = {'x': df.to_numpy()[:, :-2], 'y' : df.to_numpy()[:, -1]}
	with open(DIR + 'test/' + str(i) + '.npz', 'wb') as f:
		np.savez_compressed(f, data=test_data)