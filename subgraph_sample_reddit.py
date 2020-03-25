import numpy as np
import random
import scipy
import scipy.sparse as sp
import torch
import pickle
import networkx as nx

from utils_reddit import normalize_adj


def subgraph_sample(graph, idx_train, idx_val, idx_test, G):#input一个邻接矩阵、 设置的子图size，output一批已经采样好的子图 
    #num_init_node,size are hyperparameters
    size_of_graph = 232965

    target_size = 10000#(size_of_graph//100) + 1
    number_subgraph = 100

    num_train = len(idx_train)
    num_val = len(idx_val)
    num_test = len(idx_test)
    
    subgraph_set_dict = {}
    print('Target number of subgraph:', number_subgraph)
    for iter in range(number_subgraph):
        #select initial node
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
                if ((target_size - len_subgraph) > (len_neighbors/2)):
                    neig_random = random.sample(neighbors, max(1, len_neighbors//2))
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
        index_subgraph = list(set(index_subgraph + list(idx_val)))
        for i in range(len(index_subgraph)):
            if(index_subgraph[i] < num_train):
                idx_train_subgraph.append(i)
            elif(index_subgraph[i] < (num_train + num_val)):
                idx_val_subgraph.append(i)
            elif(index_subgraph[i] < (num_train + num_val + num_test)):
                idx_test_subgraph.append(i)
                idx_test_dict[i] = index_subgraph[i]

        print('tra val test len', len(idx_train_subgraph), len(idx_val_subgraph), len(idx_test_dict))

        #generate the adjacency matrix of the subgraph
        g = G.subgraph(index_subgraph)
        adj =nx.adjacency_matrix(g)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        index_subgraph = torch.LongTensor(index_subgraph)
        idx_train_subgraph = torch.LongTensor(idx_train_subgraph)
        idx_val_subgraph = torch.LongTensor(idx_val_subgraph)
        idx_test_subgraph = torch.LongTensor(idx_test_subgraph)

        print(iter + 1, 'th subgraph saved, the size is', index_subgraph.size())
        subgraph_set_dict[iter] = {'index_subgraph': index_subgraph, 'adj':adj,
                 'idx_train': idx_train_subgraph, 'idx_val': idx_val_subgraph,
                 'idx_test': idx_test_subgraph, 'idx_test_dict':idx_test_dict}

    return subgraph_set_dict


#adj, features, labels, idx_train, idx_val, idx_test = load_data()

y = int(232965*0.66)
idx_train = range(y)
idx_val = range(y, y+int(232965*0.005))
idx_test = range(y+int(232965*0.005), int(232965*0.72))

G = scipy.sparse.load_npz('./reddit_data/adj_full.npz').astype(np.bool)

graph = {}
for i in range(232965):
    graph[i] = list(G[i].indices)

G = G.astype(np.bool)
G = nx.from_scipy_sparse_matrix(G)


sampled_subgraph_dict = subgraph_sample(graph,idx_train,idx_val, idx_test, G)


f = open('./sampled_subgraph/reddit_ripplewalk_10000_num100', 'wb')
pickle.dump(sampled_subgraph_dict, f)
f.close()
