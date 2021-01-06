import argparse
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV



def readArguments():
	'''Auxiliary function to parse the arguments passed to the script.'''
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--num_threads', '-t', type=int, default=4, help='Number of threads for parallel computing.')
	parser.add_argument(
		'--seed', '-s', type=int, default=347, help='Seed for random K-Fold splitting.')
	parser.add_argument(
		'--num_folds', '-k', type=int, default=8, help='Number of folds.')
	parser.add_argument(
		'--dataset', '-df', type=str, default='IMDBBINARY', help='Name of the dataset.')
	parser.add_argument(
		'--output_directory', '-o', type=str, default='../data/GNTKs', help='Path to output directory.')
	return parser.parse_args()


def splitData(labels, num_folds, seed):
	''''''
	n = len(labels)
	train_indices = []
	test_indices = []
	# Init the splitter
	skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
	# Providing labels is sufficient to generate splits
	for train_idx, test_idx in skf.split(np.zeros(n), labels):
		train_indices.append(train_idx)
		test_indices.append(test_indices)
	return train_indices, test_indices


def main():
	# Read the script parameters
	args = readArguments()
	# Load the kernel matrix and the labels for the given dataset
	gram = np.load(f'{args.output_directory}/{args.dataset}/gram.npy')
	gram = gram / (gram.min() + 1e-06)
	labels = np.load(f'{args.output_directory}/{args.dataset}/labels.npy')
	# Generate k-fold cross validation train/test splits
	train_indices, test_indices = splitData(labels, args.num_folds, args.seed)
	# C values grid
	C_list = [0.001]#np.logspace(0, 1, 2)
	svc = SVC(kernel='precomputed', cache_size=10, max_iter=3e05)
	clf = GridSearchCV(
		svc, {'C': C_list}, cv=zip(train_indices, test_indices), 
		n_jobs=args.num_threads, verbose=3, return_train_score=True)
	# Fit the model
	clf.fit(gram, labels)
	# Prettify final results
	results = pd.DataFrame({
		'C': C_list,
		'train': clf.cv_results_['mean_train_score'],
		'test': clf.cv_results_['mean_test_score']
	})

	print(results)

	 


if __name__ == '__main__':
	main()
