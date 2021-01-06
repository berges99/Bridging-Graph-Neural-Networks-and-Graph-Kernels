
import numpy as np
import scipy as sp



class GNTK:
	'''
	Object class for computing the Graph Neural Tangent Kernel for any two given graphs G and G'.

	'''
	def __init__(self, 
		         num_block_operations, 
		         num_fc_layers, 
		         readout_operation, 
		         scaling_factor):
		''''''
		self.num_block_operations = num_block_operations
		self.num_fc_layers = num_fc_layers
		self.readout_operation = readout_operation
		self.scaling_factor = scaling_factor
		# Assert the given parameters are valid
		assert self.readout_operation in ['sum', 'jkn'], \
			'Readout operation must be one of ["sum", "jkn"]'
		assert self.scaling_factor in ['uniform', 'degree'], \
			'Scaling factor must be one of ["uniform", "degree"]'


	def __neighborhoodAggregation(self, aggr_step, adjacency_relation, scaling_v):
		'''Perform the neighborhood aggregation step in the beginning of each block operation.

		Parameters
			- aggr_step: (np.ndarray) covariance or gntk matrix to be updated
			- adjacency_relation: (sparse matrix) adjacency relation between G1 and G2
			- scaling_v: (np.array) vector with all scaling factors (1/(c_v*c_v'))

		Returns
			- (np.ndarray) updated covariance matrix after neighborhood aggregation

		'''
		n1, n2 = aggr_step.shape
		# (n1n2 x n1n2) * (n1n2 x 1) = (n1n2 x 1) --reshape--> (n1 x n2)
		return adjacency_relation.dot(aggr_step.reshape(-1)).reshape(n1, n2) * scaling_v


	def __updateCovariances(self, sigma, D1=None, D2=None):
		'''Feed forward through one normal layer for all elements.

		Parameters
			- sigma: (np.ndarray) covariances of last layer
			- D1, D2: (np.ndarray)

		Returns:
			- (np.ndarray) updated sigma matrix
			- (np.ndarray) updated dot sigma matrix

		'''
		# If no diagonal elements are provided
		if D1 is None:
			D1 = np.sqrt(np.diag(sigma))
			D2 = np.copy(D1)
		sigma = sigma / np.expand_dims(D1, axis=1) / np.expand_dims(D2, axis=0)
		# Ensure covariance values are bounded
		sigma = np.clip(sigma, -1, 1)
		dot_sigma = (np.pi - np.arccos(sigma)) / np.pi
		sigma = (sigma * (np.pi - np.arccos(sigma)) + np.sqrt(1 - sigma * sigma)) / np.pi
		sigma = sigma * np.expand_dims(D1, axis=1) * np.expand_dims(D2, axis=0)
		return sigma, dot_sigma, D1


	def diag(self, G, A):
		'''Compute the GNTK diagonal element for graph G with adjacency matrix A.

		Parameters:
			- G: (networkx graph) input graph
			- A: (sparse matrix) adjacency matrix of G

		Returns:
			- (list) list with all diagonal elements for all layers

		'''
		diag_list = []
		# Set scaling factor (c_v * c_v') for neighborhood aggregation step
		if self.scaling_factor == 'uniform':
			scaling_v = 1.0
		else:
			scaling_v = 1.0 / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))
		# Compute neighborhood aggregation pulses with the kronecker product
		# of sparse matrices A with itself
		adjacency_relation = sp.sparse.kron(A, A)
		# Init covariance matrix ang gntk matrices as input covariances between node features
		sigma = np.matmul(G.node_features, G.node_features.T)
		ntk = np.copy(sigma)
		# For all block operations
		for n_block in range(1, self.num_block_operations):
			# Perform neighborhood aggregation step
			sigma = self.__neighborhoodAggregation(sigma, adjacency_relation, scaling_v)
			ntk = self.__neighborhoodAggregation(ntk, adjacency_relation, scaling_v)
			# Apply L transformations through the fully-connected layers
			for fc_layer in range(self.num_fc_layers):
				sigma, dot_sigma, d = self.__updateCovariances(sigma)
				diag_list.append(d)
				ntk = ntk * dot_sigma + sigma
		return diag_list


	def gntk(self, G1, G2, A1, A2, D1_list, D2_list):
		'''Calculate the GNTK value Theta(G1, G2).

		Parameters:
			- G1, G2: (networkx graphs) input graphs
			- A1, A2: (sparse matrices) adjacency matrices of G1 and G2 respectively
			- D1_list, D2_list: (lists) lists with diagonal GNTK elements of G1 and G2

		Returns:
			- (float) Graph Neural Tangent Kernel value between the two input graphs

		'''
		# Set scaling factor (c_v * c_v') for neighborhood aggregation step
		if self.scaling_factor == 'uniform':
			scaling_v = 1.0
		else:
			scaling_v = 1.0 / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))
		# Compute neighborhood aggregation pulses with the kronecker product
		# of sparse matrices A1 and A2
		adjacency_relation = sp.sparse.kron(A1, A2)
		# With jumping knowledge
		jump_ntk = 0
		# Init covariance matrix ang gntk matrices as input covariances between node features
		sigma = np.matmul(G1.node_features, G2.node_features.T)
		ntk = np.copy(sigma)
		# Add block 0 tangent kernel
		jump_ntk += ntk
		# For all block operations
		for n_block in range(1, self.num_block_operations):
			# Perform neighborhood aggregation step
			sigma = self.__neighborhoodAggregation(sigma, adjacency_relation, scaling_v)
			ntk = self.__neighborhoodAggregation(ntk, adjacency_relation, scaling_v)
			# Apply L transformations through the fully-connected layers
			for fc_layer in range(self.num_fc_layers):
				index = (n_block - 1) * self.num_fc_layers + fc_layer
				# Update covariance matrices
				sigma, dot_sigma, _ = \
					self.__updateCovariances(sigma, D1_list[index], D2_list[index])
				# Update the gntk values
				ntk = ntk * dot_sigma + sigma
			# Add block n_block tangent kernel
			jump_ntk += ntk
		# Return the final NTK scaled by c_sigma = 2
		return np.sum(ntk) * 2 if self.readout_operation == 'sum' else np.sum(jump_ntk) * 2
