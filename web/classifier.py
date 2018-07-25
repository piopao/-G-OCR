import sys
from FCNeuralNet import *
from Solver import *
import numpy as np
import time
from read_data import read_data
from collections import Counter



class Classifier(object):

	def __init__(self, load_folder_name = 'save'):
		val = 'test'

		if val == 'test':

			#self.solver = Solver(mode='usage',load_folder_name = 'save')
			self.solver2 = Solver(mode='usage',load_folder_name = 'save - 300-300-300 1')
			self.solver3 = Solver(mode='usage',load_folder_name = 'save - 200-200-200 2')
			self.solver4 = Solver(mode='usage',load_folder_name = 'save - 300-300-300 2')
			self.solver5 = Solver(mode='usage',load_folder_name = 'save - 500-500')
			self.solver6 = Solver(mode='usage',load_folder_name = 'save - 500-2000-500-200')
			self.solver7 = Solver(mode='usage',load_folder_name = 'save - 1000-500-200')
			self.solver8 = Solver(mode='usage',load_folder_name = 'save - 2500-1000-500-200')
			self.solver9 = Solver(mode='usage',load_folder_name = 'save - 200-200-200 1')
			self.solver10 = Solver(mode='usage',load_folder_name = 'save - 200-500-200')
			

		elif val == 'train':
			start_time = time.time()
			data = read_data(['d5','d2','d4','d7','d8','d9','d11','d12','d13','arial','l1','l2','l3','l4','l5','l6','d6','d3','d10'])
			data['X_train'] = data['X_train']
			data['y_train'] = data['y_train']
			for k, v in list(data.items()):
				print(('%s: ' %k, v.shape))

			model = FCNeuralNet([200,200],input_dim=20*10, num_classes=33,
			              weight_scale=1e-2, reg=0.05)
			solver = Solver(model = model, data =data, num_epochs=100, batch_size=32,
		                learning_rate = 1e-3,
		                lr_decay = 1,
		                useless_epochs = 5
		         )
			solver.train()
			solver.save_NN()
			
			print('Part one done.')


			model = FCNeuralNet( load_folder_name = 'save')
			solver = Solver(model = model, data =data, num_epochs=500, batch_size=2048, #data['X_train'].shape[0]
		                learning_rate = 1e-4,
		                lr_decay = 1,
		                useless_epochs = 20
		         )
			solver.train()
			solver.save_NN()

			elapsed_time = time.time() - start_time
			print(elapsed_time)

	def classify_image(self, X):
		y_pool = []
		y_pool.append(self.solver2.classify(X))
		y_pool.append(self.solver3.classify(X))
		y_pool.append(self.solver4.classify(X))
		y_pool.append(self.solver5.classify(X))
		y_pool.append(self.solver6.classify(X))
		y_pool.append(self.solver7.classify(X))
		y_pool.append(self.solver8.classify(X))
		y_pool.append(self.solver9.classify(X))
		y_pool.append(self.solver10.classify(X))
		cnt = Counter(y_pool)
		return cnt.most_common(1)[0][0]