import torch
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GCNConv, GINConv
import torch.nn.functional as F
from typing import Optional
import networkx as nx
from graphxai.base_explainer import _BaseDecomposition
from graphxai.explainer_visualization import to_networkx_conv, match_torch_to_nx_edges
import matplotlib.pyplot as plt


class Explanation:
    """
    Members:
        feature_imp (torch.Tensor): Feature importance scores
            - Size: (x1,) with x1 = number of features
        node_imp (torch.Tensor): Node importance scores
            - Size: (n,) with n = number of nodes in subgraph or graph
        edge_imp (torch.Tensor): Edge importance scores
            - Size: (e,) with e = number of edges in subgraph or graph
        node_idx (int): Index for node explained by this instance
        node_reference (tensor of ints): Tensor matching length of `node_reference`
            which maps each index onto original node in the graph
        edge_reference (tensor of ints): Tensor maching lenght of `edge_reference`
            which maps each index onto original edge in the graph's edge
            index
        graph (torch_geometric.data.Data): Original graph on which explanation
            was computed
            - Optional member, can be left None if graph is too large
    Optional members:
        enc_subgraph (Subgraph): k-hop subgraph around
            - Corresponds to nodes and edges comprising computational graph around node
    """

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

        G = to_networkx_conv(self.graph, to_undirected=True)

        draw_args = dict()

        # Node weights defined by node_imp:
        if self.node_imp is not None:
            # Get node weights
            # print('node imp shape', self.node_imp.shape)
            # print('num nodes', len(list(G.nodes())))
            if isinstance(self.node_imp, torch.Tensor):
                node_imp_heat = [agg_nodes(self.node_imp[n]).item() for n in G.nodes()]
                # node_imp_map = {i:self.node_imp[i].item() for i in range(G.number_of_nodes())}
            else:
                node_imp_heat = [agg_nodes(self.node_imp[n]) for n in G.nodes()]
                # node_imp_map = {i:self.node_imp[i] for i in range(G.number_of_nodes())}

            draw_args["node_color"] = node_imp_heat

            atom_list = ["C", "N", "O", "F", "I", "Cl", "Br"]
            atom_map = {i: atom_list[i] for i in range(len(atom_list))}
            atoms = []
            for i in range(self.graph.x.shape[0]):
                atoms.append(atom_map[self.graph.x[i, :].tolist().index(1)])
            draw_args["labels"] = {i: atoms[i] for i in range(len(G.nodes))}

        # Don't do anything for feature imp
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, ax=ax, **draw_args)
        # nx.draw_networkx_labels(G, pos, labels=map, ax=ax)

        if show:
            plt.show()

        return G, pos


class CAM(_BaseDecomposition):
    """
    Class-Activation Mapping for GNNs
    """

    def __init__(self, model: torch.nn.Module, activation=None):
        """
        .. note::
            From Pope et al., CAM requires that the layer immediately before the softmax layer be
            a global average pooling layer, or in the case of node classification, a graph convolutional
            layer. Therefore, for this algorithm to theoretically work, there can be no fully-connected
            layers after global pooling. There is no restriction in the code for this, but be warned.

        Args:
            model (torch.nn.Module): model on which to make predictions
            activation (method, optional): activation funciton for final layer in network. If `activation = None`,
                explainer assumes linear activation. Use `activation = None` if the activation is applied
                within the `forward` method of `model`, only set this parameter if another activation is
                applied in the training procedure outside of model. (:default: :obj:`None`)
        """
        super().__init__(model=model)
        self.model = model

        # Set activation function
        self.activation = (
            lambda x: x if activation is None else activation
        )  # i.e. linear activation if none provided

    def get_explanation_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: int = None,
        num_nodes: int = None,
        forward_kwargs: dict = {},
    ) -> Explanation:
        """
        Explain one graph prediction by the model.

        Args:
            x (torch.Tensor): Tensor of node features from the graph.
            edge_index (torch.Tensor): Edge_index of graph.
            label (int, optional): Label on which to compute the explanation for
                this node. If `None`, the predicted label from the model will be
                used. (default: :obj:`None`)
            num_nodes (int, optional): number of nodes in graph (default: :obj:`None`)
            forward_kwargs (dict, optional): Additional arguments to model.forward
                beyond x and edge_index. Must be keyed on argument name.
                (default: :obj:`{}`)

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`None`
                `node_imp`: :obj:`torch.Tensor, [num_nodes,]`
                `edge_imp`: :obj:`None`
                `graph`: :obj:`torch_geometric.data.Data`
        """

        N = maybe_num_nodes(edge_index, num_nodes)

        # Forward pass:
        label = (
            int(self.__forward_pass(x, edge_index, forward_kwargs).argmax(dim=1).item())
            if label is None
            else label
        )

        # Steps through model:
        walk_steps, _ = self.extract_step(
            x, edge_index, detach=True, split_fc=True, forward_kwargs=forward_kwargs
        )

        # Generate explanation for every node in graph
        node_explanations = []
        for n in range(N):
            node_explanations.append(self.__exp_node(n, walk_steps, label))

        # Set Explanation class:
        exp = Explanation(node_imp=torch.tensor(node_explanations))
        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def __forward_pass(self, x, edge_index, forward_kwargs={}):
        # Forward pass:
        self.model.eval()
        pred = self.model(x, edge_index, **forward_kwargs)

        return pred

    def __exp_node(self, node_idx, walk_steps, predicted_c):
        """
        Gets explanation for one node
        Assumes ReLU activation after last convolutiuonal layer
        TODO: Fix activation function assumption
        """
        last_conv_layer = walk_steps[-1]

        if isinstance(last_conv_layer["module"][0], GINConv):
            weight_vec = (
                last_conv_layer["module"][0].nn.weight[predicted_c, :].detach()
            )  # last_conv_layer['module'][0].lin.weight[predicted_c, :].detach()
        elif isinstance(last_conv_layer["module"][0], GCNConv):
            weight_vec = (
                last_conv_layer["module"][0].lin.weight[predicted_c, :].detach()
            )
        elif isinstance(last_conv_layer["module"][0], torch.nn.Linear):
            weight_vec = last_conv_layer["module"][0].weight[predicted_c, :].detach()

        F_l_n = F.relu(last_conv_layer["input"][node_idx, :]).detach()

        L_cam_n = F.relu(torch.matmul(weight_vec, F_l_n))

        return L_cam_n.item()
