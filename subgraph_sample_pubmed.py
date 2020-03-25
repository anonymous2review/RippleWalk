import numpy as np
import random
import scipy.sparse as sp
import sys
import torch
import pickle
import networkx as nx
from utils import normalize_adj

def subgraph_sample(graph, idx_train, idx_val, idx_test):
    '''
    graph: the original graph
    idx_train: the index of training set nodes in the graph
    idx_val: the index of validation set nodes in the graph
    idx_test: the index of test set nodes in the graph
    '''
    size_of_graph = len(graph)

    target_size = 3000
    number_subgraph = 50

    num_train = idx_train.size()[0]
    num_val = idx_val.size()[0]
    num_test = idx_test.size()[0]

    subgraph_set_dict = {}
    print('Target number of subgraph:', number_subgraph)
    for iter in range(number_subgraph):
        #select initial node, and store it in the index_subgraph list
        index_subgraph = [np.random.randint(0, num_train)]
        #the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]
        len_subgraph = 0
        while(1):
            len_neighbors = len(neighbors)
            if(len_neighbors == 0):#getting stuck in the inconnected graph, select restart node
                while(1):    
                    restart_node = np.random.randint(0, num_train)
                    if(restart_node not in index_subgraph):
                        break
                index_subgraph.append(restart_node)
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
            else:
                #select part (half) of the neighbor nodes and insert them into the current subgraph
                if ((target_size - len_subgraph) > (len_neighbors*0.5)):#judge if we need to select that much neighbors
                    neig_random = random.sample(neighbors, max(1, int(0.5*len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (target_size - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    break

        idx_train_subgraph = []
        idx_val_subgraph = []
        idx_test_subgraph = []
        idx_test_dict = {}
        index_subgraph = list(set(index_subgraph + list(range(num_train, num_train+num_val))))
        #record the new index of nodes in the subgraph
        for i in range(len(index_subgraph)):
            if(index_subgraph[i]<num_train):
                idx_train_subgraph.append(i)
            elif(index_subgraph[i]<(num_train + num_val)):
                idx_val_subgraph.append(i)
            elif(index_subgraph[i]<(num_train + num_val + num_test)):
                idx_test_subgraph.append(i)
                idx_test_dict[i] = index_subgraph[i]

        print(iter + 1, 'th subgraph has been sampled')

        #generate the adjacency matrix of the subgraph
        G = nx.from_dict_of_lists(graph)
        g = G.subgraph(index_subgraph)
        adj =nx.adjacency_matrix(g)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        index_subgraph = torch.LongTensor(index_subgraph)
        idx_train_subgraph = torch.LongTensor(idx_train_subgraph)
        idx_val_subgraph = torch.LongTensor(idx_val_subgraph)
        idx_test_subgraph = torch.LongTensor(idx_test_subgraph)

        #store the information of generated subgraph: 
        #indices of nodes in the original graph G;
        #adjacency matrix;
        #new indices (indices in the subgraph) of nodes belong to train, val, test set.
        #In this way, we do not have to load the adjacency matrix of the original graph during the training process
        subgraph_set_dict[iter] = {'index_subgraph': index_subgraph, 'adj':adj,
                 'idx_train': idx_train_subgraph, 'idx_val': idx_val_subgraph,
                 'idx_test': idx_test_subgraph, 'idx_test_dict':idx_test_dict}

    return subgraph_set_dict


def load_graph(dataset_str = 'pubmed'):
    '''
    dataset_str: the name of the dataset (default as 'pubmed')
    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    return y, graph

if __name__ == "__main__":

    dataset_name = 'pubmed'

    y, graph = load_graph(dataset_name)


    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = range(len(y) + 500, len(y) + 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    sampled_subgraph_dict = subgraph_sample(graph,idx_train,idx_val,idx_test)

    #save the sampled subgraph
    f = open('./sampled_subgraph/pubmed_ripplewalk_size3000_num50', 'wb')
    pickle.dump(sampled_subgraph_dict, f)
    f.close()
