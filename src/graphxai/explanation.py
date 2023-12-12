import os
import pickle
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from typing import Optional
from sklearn import preprocessing
from settings.config import CURRENT_DATE, DEVICE, ROOT_PATH
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class Explanation:
    def __init__(
        self,
        feature_imp: Optional[torch.tensor] = None,
        node_imp: Optional[torch.tensor] = None,
        edge_imp: Optional[torch.tensor] = None,
        graph=None,
    ):
        # Establish basic properties
        self.feature_imp = feature_imp
        self.node_imp = node_imp
        self.edge_imp = edge_imp

        self.graph = graph

    def generate_masked_graph(
        self, y_pred, y_masked_pred=None, threshold=0, is_important_mask=None
    ) -> Data:
        important_masked_graph = self.graph.clone().to(DEVICE)
        unimportant_masked_graph = self.graph.clone().to(DEVICE)

        important_masked_node_idx = [
            i for i, v in enumerate(self.node_imp) if v > threshold
        ]
        uninmportant_masked_node_idx = [
            i for i, v in enumerate(self.node_imp) if v <= threshold
        ]

        important_masked_graph.x[important_masked_node_idx] = 0
        unimportant_masked_graph.x[uninmportant_masked_node_idx] = 0

        important_masked_graph = Data(
            x=important_masked_graph.x,
            edge_index=important_masked_graph.edge_index,
            y=important_masked_graph.y,
            **{"y_pred": y_pred, "y_masked_pred": None},
        )

        unimportant_masked_graph = Data(
            x=unimportant_masked_graph.x,
            edge_index=unimportant_masked_graph.edge_index,
            y=unimportant_masked_graph.y,
            **{"y_pred": y_pred, "y_masked_pred": None},
        )

        return important_masked_graph, unimportant_masked_graph

    def save_masked_graph(self, masked_graphs: list[Data], filename: str):
        path = f"{ROOT_PATH}/data/MUTAG/masked_graphs/{CURRENT_DATE}/"
        file_name = f"{filename}.pkl"
        try:
            if os.path.exists(path):
                with open(path + file_name, "wb") as f:
                    pickle.dump(masked_graphs, f)

            if not os.path.exists(path):
                os.makedirs(path)
                with open(path + file_name, "wb") as f:
                    pickle.dump(masked_graphs, f)
            # print(f"Masked graphs '{file_name}' is saved")
        except Exception as e:
            print("Error saving masked graphs: ", e)

    def set_whole_graph(self, data: Data):
        """
        Args:
            data (torch_geometric.data.Data): Data object representing the graph to store.

        :rtype: :obj:`None`
        """
        self.graph = data

    def visualize_graph(
        self, ax=None, show=False, agg_nodes=torch.mean, use_node_importance=False
    ):
        """
        Draws the graph of the Explanation

        """

        if ax is None:
            ax = plt.gca()

        G = self.__to_networkx_conv(self.graph, to_undirected=True)

        draw_args = dict()
        atom_list = ["C", "N", "O", "F", "I", "Cl", "Br"]
        atom_map = {i: atom_list[i] for i in range(len(atom_list))}
        atoms = []
        for i in range(self.graph.x.shape[0]):
            try:
                atoms.append(atom_map[self.graph.x[i, :].tolist().index(1)])
            except ValueError as e:
                atoms.append("")
        draw_args["labels"] = {i: atoms[i] for i in range(len(G.nodes))}
        draw_args["node_color"] = [0 for n in G.nodes()]

        # Node weights defined by node_imp:
        if self.node_imp is not None and use_node_importance:
            if isinstance(self.node_imp, torch.Tensor):
                node_imp_heat = [agg_nodes(self.node_imp[n]).item() for n in G.nodes()]
            else:
                node_imp_heat = [agg_nodes(self.node_imp[n]) for n in G.nodes()]

            draw_args["node_color"] = node_imp_heat

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
        if use_node_importance:
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
        prediction: int,
        label: int,
    ) -> Explanation:
        final_conv_acts = self.model.final_conv_acts
        final_conv_grads = self.model.final_conv_grads
        model_output_weights = self.model.output.weight
        node_explanations = self.__cam(
            final_conv_acts, model_output_weights, prediction
        )

        # Set Explanation class:
        exp = Explanation(node_imp=torch.tensor(node_explanations))
        exp.set_whole_graph(Data(x=x, edge_index=edge_index, y=label))

        return exp

    def __cam(self, final_conv_acts, model_output_weights, prediction):
        node_heat = torch.matmul(
                            final_conv_acts,
                            model_output_weights[prediction],
                        )
        
        #normalized_node_heat_map = preprocessing.normalize([node_heat.detach().numpy()], norm="l1").tolist()[0]
        #normalized_node_heat_map = F.normalize(node_heat, dim=0)
        min_val = torch.min(node_heat)
        max_val = torch.max(node_heat)
        normalized_node_heat_map = (node_heat - min_val) / (max_val - min_val + 1e-16)
        
        return normalized_node_heat_map