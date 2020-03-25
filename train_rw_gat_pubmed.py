from __future__ import division
from __future__ import print_function
import pickle
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils_citeseer_pubmed import load_data, accuracy
from models import GAT

# Training settings
#torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
features, labels = load_data('pubmed')



# Model and optimizer

model = GAT(nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()

features, labels = Variable(features), Variable(labels)

def train(epoch, features, adj, labels, idx_train, idx_val):
    t = time.time()
    if args.cuda:
        adj = adj.cuda()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), loss_train.data.item(), acc_train.data.item(), acc_val.data.item()


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 1

#load ripple walk subgraphs
destination_folder_path = './sampled_subgraph/'
destination_file_name_g = 'pubmed_ripplewalk_size3000_num50'
g = open(destination_folder_path + destination_file_name_g, 'rb')
content_g = pickle.load(g)
g.close()

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
iter = 0
for epoch in range(args.epochs):
    for subgraph in content_g:
        iter += 1
        index = content_g[subgraph]['index_subgraph']
        idx_train = content_g[subgraph]['idx_train']
        idx_val = content_g[subgraph]['idx_val']
        adj = content_g[subgraph]['adj']
        adj = torch.FloatTensor(np.array(adj.todense()))
        val_loss, train_loss, train_acc, val_acc = train(iter,features[index],adj,labels[index],
            idx_train, idx_val)
        
        loss_values.append(val_loss)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        torch.save(model.state_dict(), '{}.pkl'.format(iter))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = iter
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
acc_sum = 0
loss_sum = 0
with torch.no_grad():
    model.eval()
    for subgraph in content_g:
        index = content_g[subgraph]['index_subgraph']
        idx_test = content_g[subgraph]['idx_test']
        adj = content_g[subgraph]['adj']
        adj = torch.FloatTensor(np.array(adj.todense()))
        if args.cuda:
            adj = adj.cuda()
        output = model(features[index], adj)
        labels_test = labels[index]
        loss_test = F.nll_loss(output[idx_test], labels_test[idx_test])
        acc_test = accuracy(output[idx_test], labels_test[idx_test])
        acc_sum += acc_test
        loss_sum += loss_test
        #idx_test_dict = content_g[subgraph]['idx_test_dict']
        #for i in idx_test_dict:
        #    test_f1_score[idx_test_dict[i]].append(output[i])

# f1_sum = 0
# acc_sum = 0
num_test = len(content_g)
# num_null_test = 0
# for i in test_f1_score:
#     len_test = len(test_f1_score[i])
#     if(len_test== 0):
#         num_null_test += 1
#         continue
#     else:
#         num_test += 1
#         test_i = np.sum(test_f1_score[i], axis=0)/len_test
#         preds_test = test_i == max(test_i)#test_i.max(1)[1].type_as(labels[i])
#         f1_sum += f1_score(labels[i].cpu().detach().numpy(),
#             preds_test.cpu().detach().numpy(), average='micro')
#         if(preds_test == labels[i]):
#             acc_sum += 1

print("Test set results:",
          "loss= {:.4f}".format(loss_sum/num_test),#data[0]),
          "acc= {:.4f}".format(acc_sum/num_test))#data[0])),
          #"acc= {:.4f}".format(acc_test))#data[0]))

data = {'train_loss': train_loss_list, 'train_acc': train_acc_list,
        'val_loss': val_loss_list, 'val_acc': val_acc_list,
        'test_acc': acc_sum/num_test, 'test_loss': loss_sum/num_test}

f = open('./results/pubmed_gat_ripplewalk_size3000_num50', 'wb')
pickle.dump(data, f)
f.close()
