# from past.builtins import xrange
import os
from FCNeuralNet import *

import numpy as np

class Solver(object):

    def __init__(self, mode='train', model = None, load_folder_name = 'save', data = None, 
            learning_rate = 1e-2, lr_decay = 1.0, batch_size = 128, num_epochs = 5):
        self.ANBANI = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']

        if mode == 'usage':
            self.load_NN(load_folder_name)
            return

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.optim_config = {'learning_rate' : learning_rate}
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}

        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    #TODO: save folder name
    def save_NN(self):
        self.model.save()


    def load_NN(self, load_folder_name):
        self.model = FCNeuralNet( load_folder_name = load_folder_name)


    def sgd_momentum(self, w, dw, config=None):
        if config is None: config = {}
        config.setdefault('learning_rate', 1e-2)
        config.setdefault('momentum', 0.9)
        v = config.get('velocity', np.zeros_like(w))

        next_w = None
        v = config['momentum'] * v - config['learning_rate'] * dw
        next_w = w + v
        config['velocity'] = v

        return next_w, config

    def step(self):
        num_train = self.X_train.shape[0]
        batch_index = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_index]
        y_batch = self.y_train[batch_index]

        loss, grads = self.model.loss(X_batch, y_batch)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.sgd_momentum(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def classify(self, X):
        scores = self.model.loss(X)
        y_pred = np.argmax(scores,axis=1)
        return self.ANBANI[y_pred[0]]

   
    def check_acc(self, X, y, num_samples = None, batch_size = 128):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        y_pred2 = []
        y_pred_percent = []
        y_pred2_percent = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
            y_pred2.append(np.argsort(scores,axis=1)[:,-2])

        y_pred = np.hstack(y_pred)
        y_pred2 = np.hstack(y_pred2)
        acc = np.mean(y_pred == y)

        if batch_size == 1:
            accarr = np.zeros((33,2))
            for i in range(y.shape[0]):
                if y_pred[i] == y[i]:
                    accarr[y[i]][0] += 1
                accarr[y[i]][1] += 1
            for i in range(accarr.shape[0]):
                print(i,' ',accarr[i][0],' ',accarr[i][1],' ',accarr[i][0]/max(accarr[i][1],1))
            print('sum: ',accarr.sum(axis=0))
            accexact = {}
            secondguess = {}
            for i in range(y.shape[0]):
                if y[i] != y_pred[i]:
                    if y[i] not in accexact:
                        accexact[y[i]] = []
                        secondguess[y[i]] = []
                    accexact[y[i]].append(y_pred[i])
                    secondguess[y[i]].append(y_pred2[i])
            for i in range(33):
                if i in accexact:
                    print(i,' ',accexact[i],secondguess[i])
        return acc


    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(int(num_train / self.batch_size), 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        useless_epoch_cnt = 0
        for t in xrange(num_iterations):
            #if we don't have eny progress
            if useless_epoch_cnt > self.useless_epochs:
                print('No more progress')
                self.model.params = self.best_params
                return
            self.step()

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

                train_acc = self.check_acc(self.X_train, self.y_train,
                    num_samples = 1000)
                val_acc = self.check_acc(self.X_val, self.y_val)
                test_acc = self.check_acc(self.X_test, self.y_test)

                epoch_num = (t+1)/iterations_per_epoch
                print('epoch: ', epoch_num,' useless streak: ',useless_epoch_cnt)
                print('\n accs: ',train_acc,' ',val_acc,' ',test_acc)

                if val_acc > self.best_val_acc:
                    useless_epoch_cnt = 0
                    self.best_val_acc = val_acc
                    # since we have new best parameters, lets copy
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
                else:
                    useless_epoch_cnt += 1

        self.model.params = self.best_params
