import os
import tarfile
import numpy as np 
import imp
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import train_test_split
from six.moves import urllib


import hashlib

def test_set_check(identifier, test_ratio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test(data, test_ratio, id_column, hash = hashlib.md5):
	pass


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	if not os.path.isdir(housing_path):
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]



# Testing

Task = namedtuple('Task', ['summary', 'owner', 'done', 'id'])
Task.__new__.__defaults__ = (None, None, False, None)

def test_functions():
	pass

def test_import():
	try:
		imp.find_module('tarfile')
		found = True
	except ImportError:
		found = False
	assert found == True
	

def test_asdict():
	"""
	as_dict() should return a dictionary
	"""
	t_task = Task("do something", "okken", True, 21)
	t_dict = t_task._asdict()
	expected = {'summary': 'do something',
				'owner' : 'okken',
				'done' : True,
				'id': 21}
	assert t_dict == expected
	
	

