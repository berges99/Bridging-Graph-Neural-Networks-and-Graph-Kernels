import random
import numpy as np
import networkx as nx





def loadData(dataset):
	'''
	TBD

	Parameters:
		-

	Returns:
		-

	'''
	print('Loading data...')

	with open(f'{dataset}/{dataset}_A.txt')