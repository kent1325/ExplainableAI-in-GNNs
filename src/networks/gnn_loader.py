import torch
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, feature_size, num_of_classes):
        super(GCN, self).__init__()
        embedding_size = 7

        # Explanation variables
        self.final_conv_acts = None
        self.final_conv_grads = None

        # GCN Layers
        self.input = GCNConv(feature_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.output = Linear(embedding_size, num_of_classes)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch_index):
        out = self.input(x, edge_index)
        out = F.relu(out)
        with torch.enable_grad():
            self.final_conv_acts = F.relu(self.conv1(out, edge_index))
        self.final_conv_acts.register_hook(self.activations_hook)
        out = gap(out, batch_index)
        out = self.output(out)
        out = F.softmax(out, dim=1)

        return out