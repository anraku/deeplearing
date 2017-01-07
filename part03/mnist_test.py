import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

# 取得したデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(y_test.shape)
