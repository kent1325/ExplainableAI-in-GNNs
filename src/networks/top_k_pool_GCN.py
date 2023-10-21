import torch
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear
import torch.nn.functional as F


class GCN_pool_layers(torch.nn.Module):
    def __init__(self, feature_size):
        super(GCN_pool_layers, self).__init__()
        #torch.manual_seed(12345)
        embedding_size = 32

        # GCN Layers
        self.input = GCNConv(feature_size, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.output = Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch_index):
        out = self.input(x, edge_index)
        out = F.relu(out)
        out, edge_index, _, batch_index, _, _ = self.pool1(out, edge_index, None, batch_index)
        out = F.relu(out)
        out = gap(out, batch_index)
        out = self.output(out)

        return out
