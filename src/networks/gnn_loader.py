import torch
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        embedding_size = 32

        # GCN Layers
        self.input = GCNConv(feature_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.output = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        out = self.input(x, edge_index)
        out = F.relu(out)
        out = self.conv1(out, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.conv3(out, edge_index)
        out = F.relu(out)

        out = torch.cat([gmp(out, batch_index), gap(out, batch_index)], dim=1)

        out = self.output(out)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GAT(torch.nn.Module):
    def __init__(self, feature_size):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        num_heads = 3
        embedding_size = 32

        # GAT layers
        self.input = GATConv(feature_size, embedding_size, heads=num_heads, dropout=0.3)
        self.conv1 = GATConv(
            embedding_size, embedding_size, heads=num_heads, dropout=0.3
        )
        self.conv2 = GATConv(
            embedding_size, embedding_size, heads=num_heads, dropout=0.3
        )

        # Tranformer layers
        self.head_transform = Linear(embedding_size * num_heads, embedding_size)

        # Pooling layers
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        self.pool3 = TopKPooling(embedding_size, ratio=0.2)

        # Linear layers
        self.linear1 = Linear(embedding_size * 2, embedding_size)
        self.output = Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch_index):
        out = self.input(x, edge_index)
        out = self.head_transform(out)
        out, edge_index, edge_attr, batch_index, _, _ = self.pool1(
            out, edge_index, None, batch_index
        )

        out1 = torch.cat([gmp(out, batch_index), gap(out, batch_index)], dim=1)

        out = self.conv1(out, edge_index)
        out = self.head_transform(out)
        out, edge_index, edge_attr, batch_index, _, _ = self.pool2(
            out, edge_index, None, batch_index
        )

        out2 = torch.cat([gmp(out, batch_index), gap(out, batch_index)], dim=1)

        out = self.conv2(out, edge_index)
        out = self.head_transform(out)
        out, edge_index, edge_attr, batch_index, _, _ = self.pool3(
            out, edge_index, None, batch_index
        )

        out3 = torch.cat([gmp(out, batch_index), gap(out, batch_index)], dim=1)

        out = out1 + out2 + out3

        out = self.linear1(out).relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.output(out)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
