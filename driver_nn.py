import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from f_load_dataset import load_dataset

# load dataset
train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()

# create l-dim network by just adding num of neurons in layer_dim
# first and last elements represent input and output layers dim
layer_dim = [1, 100, 100, 1]

# add activation functions name here. 
# input layer activation function is None
activations = [None, 'tanh', 'tanh', 'identity']
assert len(layer_dim) ==  len(activations), "layer dim or activation is missing.."

# hyper parameters of neural network
learning_rate = 1e-3
num_epochs = 10000
mini_batch_size = 10

# train neural network 
train(train_x, train_t, val_x, val_t, learning_rate, num_epochs, mini_batch_size, layer_dim, activations)


# test neural network 
train_loss, _ = test(train_x, train_t, layer_dim, activations)
print("training loss..", np.round(train_loss.data.numpy() , 4))
test_loss, test_output = test(test_x, test_t, layer_dim, activations)
print("testing loss..", np.round(test_loss.data.numpy() , 4))


# plot results 
fig = plt.gcf()
plt.plot(test_x.T, test_output.data.numpy(), linewidth=3, color='red', label="output")
plt.plot(test_x.T, test_t.T, linewidth=3, color='blue', linestyle='dashed', label="target")
plt.title('function 1')
plt.legend()
plt.grid()
plt.show()
fig.savefig('function1_results.png')
