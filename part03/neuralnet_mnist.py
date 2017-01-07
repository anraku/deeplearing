import numpy as np
import pickle
from dataset.mnist import load_mnist

# mnistデータからテストデータを取得
def get_data():
	(x_train, t_train), (x_test, t_test) = \
	load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

# pklファイルから学習データを取得
def init_network():
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)

	return network

# 入力から出力値を求めるニューラルネットの計算
def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)

	return y

# 各層で使われる活性化関数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 出力層の計算に使われる関数
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)# オーバーフロー対策
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
# テストデータを使ってニューラルネットの出力の正確さを求める
for i in range(0, len(x), batch_size):
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	# y = predict(network, x[i])
	p = np.argmax(y_batch, axis=1) # 最も確率の高い要素のインデックスを取得
	accuracy_cnt += np.sum(p == t[i:i+batch_size])
	# if p == t[i]:
	# 	accuracy_cnt += 1

# ニューラルネット出力の正答率を算出
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

