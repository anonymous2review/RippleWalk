import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolutionLayer, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv1 = GraphConvolutionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, not_final=True)
        
        self.add_module('conv1', self.conv1)

        self.conv2 = GraphConvolutionLayer(nhid, nclass, dropout=dropout, alpha=alpha, not_final=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]#multi-head
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

