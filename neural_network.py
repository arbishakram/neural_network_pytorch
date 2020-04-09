import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
import torch
import torch.nn as nn
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_loss(loss,val_loss, len_layer, learning_rate):        
    plt.figure()
    fig = plt.gcf()
    plt.plot(loss, linewidth=3, label="train")
    plt.plot(val_loss, linewidth=3, label="val")
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('learning rate =%s, hidden layers=%s' % (learning_rate, len_layer-1))
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('plot_loss.png')
        
    
def plot_gradients(net, len_layer):
    avg_l_g = []
    for idx, param in enumerate(net.parameters()):
        if idx % 2 == 0:
             weights_grad = param.grad 
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d].numpy())           
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
    layers = ['layer %s'%l for l in range(len_layer+1)]
    weights_grad_mag = avg_l_g
    fig = plt.gcf()
    plt.xticks(range(len(layers)), layers)
    plt.xlabel('layers')
    plt.ylabel('average gradients magnitude')
    plt.title('')
    plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
    plt.show() 
    fig.savefig('plot_gradients.png')
    

class NeuralNet(nn.Module):
    def __init__(self, size_list, activations):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            if activations[i+1] == 'sigmoid':
                act = nn.Sigmoid()
            elif activations[i+1] =='tanh':
                act = nn.Tanh()
            elif activations[i+1] == 'relu':
                act = nn.ReLU()
            layers.append(act)
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

def train(train_x, train_y, val_x, val_y, learning_rate, num_epochs, mini_batch_size, layer_dim, activations):
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    train_loss = []
    val_loss = []  
    num_samples = train_y.shape[1] 

    len_layer = len(layer_dim) - 1
    
    net = NeuralNet(layer_dim, activations)    
    calculate_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    print(net)
    val_x = torch.from_numpy(val_x).float().to(device)
    val_y = torch.from_numpy(val_y).float().to(device)
    
    total_parameters = sum(p.numel() for p in net.parameters())
    print("total number of parameters:", total_parameters)
    
    
    for i in range(0, num_epochs):
        for idx in range(0, num_samples, mini_batch_size):
            minibatch_input =  train_x[:, idx:idx + mini_batch_size]
            minibatch_target =  train_y[:, idx:idx + mini_batch_size]
            minibatch_input = torch.from_numpy(minibatch_input).float().to(device)
            minibatch_target = torch.from_numpy(minibatch_target).float().to(device)

            # Forward pass
            outputs = net(minibatch_input.T)
            loss = calculate_loss(outputs, minibatch_target.T)
            
            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                
              
        train_loss.append(loss)       
        outputs = net(val_x.T)
        va_loss = calculate_loss(outputs, val_y.T)            
        val_loss.append(va_loss)              
        print("Epoch %i: training loss %f, validation loss %f" % (i, loss,va_loss))
        
    plot_loss(train_loss,val_loss, len_layer, learning_rate)      
    plot_gradients(net, len_layer)
       
    
def test(test_x, test_y, layer_dim, activations):
        test_X = torch.from_numpy(test_x.T).float().to(device)
        test_y = torch.from_numpy(test_y.T).float().to(device)
        net = NeuralNet(layer_dim, activations)
        calculate_loss = nn.MSELoss()
        outputs = net(test_X)
        loss = calculate_loss(outputs, test_y)
        return loss, outputs