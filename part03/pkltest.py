import numpy as np
import pickle
from pprint import pprint
from dataset.mnist import load_mnist

def get_data():
	(x_train, t_train), (x_test, t_test) = \
	load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

# pklファイルから学習データを取得
def init_network():
	with open("./dataset/mnist.pkl", 'rb') as f:
		network = pickle.load(f)

	return network

network = init_network()

x, t = get_data()
pprint(x)
pprint(t)
