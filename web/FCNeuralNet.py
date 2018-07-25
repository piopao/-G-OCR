# from past.builtins import xrange
import numpy as np

from layers import *

class FCNeuralNet(object):
    
    def __init__(self, hidden_dims=0, input_dim=20*10, num_classes=33, reg=0.0,
                 weight_scale=1e-2, load_folder_name=None):
        if load_folder_name is not None:
            self.load(load_folder_name)
            return
        #init
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}

        layers = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(layers[i],layers[i+1])
            self.params['b'+str(i+1)] = np.zeros(layers[i+1])


    def save(self,folder_name='save'):
        np.save(folder_name + '/save_reg.npy',self.reg)
        np.save(folder_name + '/save_num_layers.npy',self.num_layers)
        for i in range(self.num_layers):
            np.save(folder_name + '/save_W'+str(i+1)+'.npy',self.params['W'+str(i+1)])
            np.save(folder_name + '/save_b'+str(i+1)+'.npy',self.params['b'+str(i+1)])
        print('saved !')


    def load(self, folder_name='save'):
        self.reg = np.load(folder_name + '/save_reg.npy')
        self.num_layers = np.load(folder_name + '/save_num_layers.npy')  
        self.params = {}
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = np.load(folder_name + '/save_W'+str(i+1)+'.npy')
            self.params['b'+str(i+1)] = np.load(folder_name + '/save_b'+str(i+1)+'.npy')
        print('loaded')


    def loss(self, X, y=None):
        scores = None
        cache = {}
        X_cur = X
        for i in range(self.num_layers-1):
            X_cur, cache['l' + str(i+1)] = affine_relu_forward(X_cur,
                                                        self.params['W'+str(i+1)],
                                                        self.params['b'+str(i+1)])
        scores, cache['l'+str(self.num_layers)] = affine_forward(X_cur,
                                                        self.params['W'+str(self.num_layers)],
                                                        self.params['b'+str(self.num_layers)])
        #it's test forward pass only
        if y is None:
            return scores

        loss, grads = 0.0, {}
        loss, dL = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]**2)

        dx, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dL, cache['l'+str(self.num_layers)])
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]

        for i in reversed(range(self.num_layers-1)):
            dx, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_relu_backward(dx, cache['l'+str(i+1)])
            grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]

        return loss, grads
