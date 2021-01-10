import os
import argparse
import scipy as sp
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

# Import user-defined packages
from gntk import GNTK
from utils import loadData



def readArguments():
	'''Auxiliary function to parse the arguments passed to the script.'''
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--num_threads', '-t', type=int, default=8, help='Number of threads for parallel computing.')
	parser.add_argument(
		'--dataset', '-df', type=str, default='IMDBBINARY', help='Name of the dataset.')
	parser.add_argument(
		'--num_block_operations', '-S', type=int, default=2, help='Number of block operations.')
	parser.add_argument(
		'--num_fc_layers', '-L', type=int, default=2, help='Number of FC layers.')
	parser.add_argument(
		'--readout_operation', '-ro', type=str, default='jkn', help='Readout operation.')
	parser.add_argument(
		'--scaling_factor', '-scale', type=str, default='degree', help='Scaling method.')
	parser.add_argument(
		'--output_directory', '-o', type=str, default='../outputs', help='Path to output directory.')
	return parser.parse_args()


def computeGNTK(indices):
	'''Auxiliary function to compute GNTK values between two given graphs G1 and G2.'''
	i, j = indices
	return gntk.gntk(
		G1=graphs[i], G2=graphs[j], 
		A1=adjacency_matrices[i], A2=adjacency_matrices[j],
		D1_list=diagonal_elements[i], D2_list=diagonal_elements[j])


def main():
	# Init global variables for the entire script
	global gntk, graphs, adjacency_matrices, diagonal_elements
	# Read the script parameters
	args = readArguments()
	# Read the graphs and get the labels
	graphs = loadData(args.dataset)
	labels = np.array([g.label for g in graphs]).astype(int)
	# Init the GNTK object
	gntk = GNTK(
		args.num_block_operations, args.num_fc_layers, 
		args.readout_operation, args.scaling_factor)
	# List with the adjacency matrices of the graphs
	adjacency_matrices = []
	# List with the diagonal covariance matrices of the graphs at all layers
	diagonal_elements = []
	print()
	print('Computing adjacency and diagonal matrices...')
	for i in tqdm(range(len(graphs))):
		n = len(graphs[i].neighbors)
		# Add self-loops -> N(v) = N(v) U {v}
		for j in range(n):
			graphs[i].neighbors[j].append(j)
		# Retrieve the edges from the graph
		edges = graphs[i].g.edges
		m = len(edges)
		# Elements for building sparse matrix in coordinate format (triplet format)
		data = [1] * m
		rows = [e[0] for e in edges]
		cols = [e[1] for e in edges]
		# Build sparse adjacency matrix for the graph g
		adjacency_matrices.append(
			sp.sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32))
		# Add self-loops to the adjacenty matrix + ensure edge bidirectionality
		adjacency_matrices[-1] = \
			adjacency_matrices[-1] + adjacency_matrices[-1].T + sp.sparse.identity(n)
		# Compute the diagonal GNTK value
		diagonal_elements.append(
			gntk.diag(graphs[i], adjacency_matrices[i]))
	# Define all graph pairs in the list
	graph_pairs = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]	
	# Init thread pool and compute GNTK between all pairs of graphs
	# Use parameter args.num_threads to specify the number of cores; use all available by default
	print()
	print('Computing all GNTK values between all pairs of graphs...')
	with Pool() as pool:
		# Use imap() function in order to enable tqdm() progress visualization.
		# Hence it can slow down somewhat the execution, substitute with map() if running time
		# gets too affected.
		gntk_values = list(tqdm(pool.imap(computeGNTK, graph_pairs), total=len(graph_pairs)))
	# Fill the symmetric kernel matrix
	gram = np.zeros((len(graphs), len(graphs)))
	for indices, gntk_value in zip(graph_pairs, gntk_values):
		i, j = indices
		gram[i, j] = gntk_value
		gram[j, i] = gntk_value
	# Save the resulting kernel matrix at the specified location
	output_name = f'{args.output_directory}/{args.dataset}'
	# Create the directory if necessary
	if not os.path.exists(output_name):
		os.mkdir(output_name)
	output_name = \
		f'{args.output_directory}/{args.dataset}/blocks{args.num_block_operations}' + \
		f'_layers{args.num_fc_layers}_{args.readout_operation}_{args.scaling_factor}'
	np.save(f'{output_name}_gram.npy', gram)
	np.save(f'{output_name}_labels.npy', labels)
	print()
	print(f'Gram matrix stored at: {output_name}_gram.npy')
	print(f'Labels stored at: {output_name}_labels.npy')


if __name__ == '__main__':
	main()
