import torch
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
from typing import Optional
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt


class Explanation:
    def __init__(
        self,
        feature_imp: Optional[torch.tensor] = None,
        node_imp: Optional[torch.tensor] = None,
        edge_imp: Optional[torch.tensor] = None,
        node_idx: Optional[torch.tensor] = None,
        node_reference: Optional[torch.tensor] = None,
        edge_reference: Optional[torch.tensor] = None,
        graph=None,
    ):
        # Establish basic properties
        self.feature_imp = feature_imp
        self.node_imp = node_imp
        self.edge_imp = edge_imp

        # Only established if passed explicitly in init, not overwritten by enclosing subgraph unless explicitly specified
        self.node_reference = node_reference
        self.edge_reference = edge_reference

        self.node_idx = node_idx  # Set this for node-level prediction explanations
        self.graph = graph

    def set_whole_graph(self, data: Data):
        """
        Args:
            data (torch_geometric.data.Data): Data object representing the graph to store.

        :rtype: :obj:`None`
        """
        self.graph = data

    def visualize_graph(self, ax=None, show=False, agg_nodes=torch.mean):
        """
        Draws the graph of the Explanation

        """

        if ax is None:
            ax = plt.gca()

        G = self.__to_networkx_conv(self.graph, to_undirected=True)

        draw_args = dict()

        # Node weights defined by node_imp:
        if self.node_imp is not None:
            if isinstance(self.node_imp, torch.Tensor):
                node_imp_heat = [agg_nodes(self.node_imp[n]).item() for n in G.nodes()]
            else:
                node_imp_heat = [agg_nodes(self.node_imp[n]) for n in G.nodes()]

            draw_args["node_color"] = preprocessing.normalize([node_imp_heat])

            atom_list = ["C", "N", "O", "F", "I", "Cl", "Br"]
            atom_map = {i: atom_list[i] for i in range(len(atom_list))}
            atoms = []
            for i in range(self.graph.x.shape[0]):
                atoms.append(atom_map[self.graph.x[i, :].tolist().index(1)])
            draw_args["labels"] = {i: atoms[i] for i in range(len(G.nodes))}

        pos = nx.kamada_kawai_layout(G)
        ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
        lc = nx.draw_networkx_labels(
            G, pos, labels=draw_args["labels"], font_weight="bold", font_color="w"
        )
        nc = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            cmap=plt.cm.viridis,
            node_size=500,
            node_color=draw_args["node_color"],
            vmin=0,
            vmax=1,
        )
        plt.colorbar(nc)
        plt.axis("off")

        if show:
            plt.show()

        return G, pos

    def __to_networkx_conv(
        self,
        data,
        node_attrs=None,
        edge_attrs=None,
        to_undirected=False,
        remove_self_loops=False,
        get_map=False,
    ):
        r"""Converts a :class:`torch_geometric.data.Data` instance to a
        :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
        a directed :obj:`networkx.DiGraph` otherwise.

        Args:
            data (torch_geometric.data.Data): The data object.
            node_attrs (iterable of str, optional): The node attributes to be
                copied. (default: :obj:`None`)
            edge_attrs (iterable of str, optional): The edge attributes to be
                copied. (default: :obj:`None`)
            to_undirected (bool, optional): If set to :obj:`True`, will return a
                a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
                undirected graph will correspond to the upper triangle of the
                corresponding adjacency matrix. (default: :obj:`False`)
            remove_self_loops (bool, optional): If set to :obj:`True`, will not
                include self loops in the resulting graph. (default: :obj:`False`)
            get_map (bool, optional): If `True`, returns a tuple where the second
                element is a map from original node indices to new ones.
                (default: :obj:`False`)
        """
        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(data.edge_index).tolist())
        map_norm = {node_list[i]: i for i in range(len(node_list))}
        # rev_map_norm = {v:k for k, v in map_norm.items()}
        G.add_nodes_from([map_norm[n] for n in node_list])

        values = {}
        for key, item in data:
            if torch.is_tensor(item):
                values[key] = item.squeeze().tolist()
            else:
                values[key] = item
            if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
                values[key] = item[0]

        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)
            for key in edge_attrs if edge_attrs is not None else []:
                G[u][v][key] = values[key][i]

        for key in node_attrs if node_attrs is not None else []:
            for i, feat_dict in G.nodes(data=True):
                feat_dict.update({key: values[key][i]})

        if get_map:
            return G, map_norm
        else:
            return G


class CAM:
    """
    Class-Activation Mapping for GNNs
    """

    def __init__(self, model: torch.nn.Module):
        # super().__init__(model=model)
        self.model = model

    def get_explanation_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: int = None,
        num_nodes: int = None,
        forward_kwargs: dict = {},
    ) -> Explanation:
        N = maybe_num_nodes(edge_index, num_nodes)

        final_conv_acts = self.model.final_conv_acts
        final_conv_grads = self.model.final_conv_grads
        node_explanations = self.__grad_cam(final_conv_acts, final_conv_grads)[:N]

        # Set Explanation class:
        exp = Explanation(node_imp=torch.tensor(node_explanations))
        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def __grad_cam(self, final_conv_acts, final_conv_grads):
        node_heat_map = []
        alphas = torch.mean(final_conv_grads, axis=0)
        for n in range(final_conv_acts.shape[0]):  # nth node
            node_heat = F.relu(alphas @ final_conv_acts[n]).item()
            node_heat_map.append(node_heat)
        return node_heat_map
