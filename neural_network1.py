import numpy as np

# 各層で使われる活性化関数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 1層目
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)
#1層目の出力
A1 = np.dot(X, W1) + B1
Z = sigmoid(A1)
print(A1)
print(Z)

# 2層目
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

# 2層目の計算
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 出力層に使われる活性化関数(恒等関数)
def identity_function(x):
	return x

# 出力層の計算
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)