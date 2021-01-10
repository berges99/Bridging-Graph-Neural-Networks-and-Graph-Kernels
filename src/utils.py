import os
import numpy as np
import networkx as nx


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    #print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('../data/dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

            
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros([len(g.node_tags), len(tagset)])
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    # print('# classes: %d' % len(label_dict))
    # print('# maximum node tag: %d' % len(tagset))

    # print("# data: %d" % len(g_list))

    return g_list, len(label_dict)



def loadData(dataset_name):
	'''
	TBD

	Parameters:
		-

	Returns:
		-

	'''
	# Social networks datasets
	if dataset_name in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
	    g_list, _ = load_data(dataset_name, True)
	# Bioinformatics datasets
	elif dataset_name in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
	    g_list, _ = load_data(dataset_name, False)
	# Elsewise the dataset is not supported (as of yet)
	else:
		raise ValueError(f'Dataset ({dataset_name}) is not supported!')
	return g_list
	# ROOT = f'../data/{dataset_name}/{dataset_name}'
	# # Row-like graph indicator
	# with open(f'{ROOT}/{dataset_name}_graph_indicator.txt', 'r') as f:
	# 	graph_indicator = [int(i) - 1 for i in list(f)]

	# ##########
	# # Nodes
	# num_graphs = max(graph_indicator)
	# node_indices = []
	# offset = []
	# c = 0
	# # Identify the row numbers pertaining to each graph
	# for i in range(num_graphs + 1):
	# 	offset.append(c)
	# 	c_i = graph_indicator.count(i)
	# 	node_indices.append((c, c + c_i - 1))
	# 	c += c_i
	# # Init all the networkx graphs
	# graph_db = []
	# for i in node_indices:
	# 	g = nx.Graph()
	# 	for j in range(i[1] - i[0] + 1):
	# 		g.add_node(j)
	# 	graph_db.append(g)

	# ##########
	# # Edges
	# with open(f'{ROOT}/{dataset_name}_A.txt', 'r') as f:
	# 	edges = [i.split(',') for i in list(f)]
	# edges = [
	# 	(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges
	# ]
	# edge_list = []
	# edgeb_list = []
	# for e in edges:
	# 	g_id = graph_indicator[e[0]]
	# 	g = graph_db[g_id]
	# 	off = offset[g_id]
	# 	#
	# 	if (e[0] - off, e[1] - off) not in list(g.edges()) and \
	# 	   (e[1] - off, e[0] - off) not in list(g.edges()):
	# 	    g.add_edge(e[0] - off, e[1] - off)
	# 	    edge_list.append((e[0] - off, e[1] - off))
	# 	    edgeb_list.append(True)
	# 	else:
	# 		edgeb_list.append(False)

	# ##########
	# # Node labels

	# ##########
	# # Node attributes

	# ##########
	# # Edge labels

	# ##########
	# # Edge attributes

	# ##########
	# # Classes
	# if os.path.exists(f'{ROOT}/{dataset_name}_graph_labels.txt'):
	# 	with open(f'{ROOT}/{dataset_name}_graph_labels.txt', 'r') as f:
	# 		classes = [i.strip() for i in list(f)]
	# 	# Allow multiple class graph labeling
	# 	classes = [i.split(',') for i in classes]
	# 	cs = []
	# 	for i, c in enumerate(classes):
	# 		cs.append([int(j.strip()) for j in c])
	# 	# Add the labels to the corresponding graphs in the main db
	# 	for i, g in enumerate(graph_db):
	# 		g.graph['classes'] = cs[i]

	# ##########
	# # Targets

	# return graph_db
