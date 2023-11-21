import torch
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


def visualize_mol_explanation(
    data: torch.Tensor,
    node_weights: list = None,
    edge_weights: list = None,
    ax: matplotlib.axes.Axes = None,
    atoms: list = None,
    weight_map: bool = False,
    show: bool = True,
    directed: bool = False,
    fig: matplotlib.figure.Figure = None,
):
    """
    Visualize explanation for predictions on a graph
    Args:
        data (torch_geometric.data): data representing the entire graph
        node_weights (list): weights by which to color the nodes in the graph
        ax (matplotlib.Axes, optional): axis on which to plot the visualization.
            If `None`, visualization is plotted on global figure (similar to plt.plot).
            (default: :obj:`None`)
        atoms (list, optional): List of atoms corresponding to each node. Used for
            node labels on the visualization. (default: :obj:`None`)
        weight_map (bool, optional): If `True`, shows node weights (literal values
            from `node_weights` argument) as the node labels on visualization. If
            `False`, atoms are used. (default: :obj:`False`)
        show (bool, optional): If `True`, calls `plt.show()` to display visualization.
            (default: :obj:`True`)
        directed (bool, optional): If `True`, shows molecule as directed graph.
            (default: :obj:`False`)
        fig (matplotlib.figure.Figure, optional): Figure for plots being drawn with this
            function. Will be used to direct the colorbar. (default: :obj:`None`)
    """
    if directed:
        G = to_networkx_conv(data, to_undirected=False, remove_self_loops=True)
    else:
        G = to_networkx_conv(data, to_undirected=True, remove_self_loops=True)

    pos = nx.kamada_kawai_layout(G)

    if node_weights is None:
        node_weights = "#1f78b4"
        map = {i: atoms[i] for i in range(len(G.nodes))}
    else:
        map = (
            {i: node_weights[i] for i in range(len(G.nodes))}
            if weight_map
            else {i: atoms[i] for i in range(len(G.nodes))}
        )

    edge_cmap = None
    edge_map = "k"
    if edge_weights is not None:
        edge_map = {i: [edge_weights[i]] for i in range(len(G.edges))}
        edge_cmap = plt.cm.Reds

    if ax is None:
        nx.draw(
            G,
            pos,
            node_color=node_weights,
            node_size=400,
            cmap=plt.cm.Blues,
            arrows=False,
            edge_color=edge_map,
            edge_cmap=edge_cmap,
        )
        nodes = nx.draw_networkx_nodes(
            G, pos, node_size=400, node_color=node_weights, cmap=plt.cm.Blues
        )

        # Set up colormap:
        if node_weights != "#1f78b4":
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.Blues,
                norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)),
            )
            plt.colorbar(sm, shrink=0.75)
        elif edge_weights is not None:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.Reds,
                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)),
            )
            plt.colorbar(sm, shrink=0.75)

        nx.draw_networkx_labels(G, pos, labels=map)

    else:
        nx.draw(
            G,
            pos,
            node_color=node_weights,
            node_size=400,
            cmap=plt.cm.Blues,
            edge_color=edge_map,
            arrows=False,
            edge_cmap=edge_cmap,
            ax=ax,
        )  # , with_labels = True)
        nx.draw_networkx_labels(G, pos, labels=map, ax=ax)

        if node_weights != "#1f78b4" and (fig is not None):
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.Blues,
                norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)),
            )
            fig.colorbar(sm, shrink=0.75, ax=ax)

        if (edge_weights is not None) and (fig is not None):
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.Reds,
                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)),
            )
            fig.colorbar(sm, shrink=0.75, ax=ax)

    if show:
        plt.show()


def to_networkx_conv(
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


def match_torch_to_nx_edges(G: nx.Graph, edge_index: torch.Tensor):
    """
    Gives dictionary matching index in edge_index to G.edges
        - Supports matching for undirected edges
        - Mainly for plotting
    """

    edges_list = list(G.edges)

    edges_map = dict()

    for i in range(len(edges_list)):
        e1, e2 = edges_list[i]

        # Check e1 -> 0, e2 -> 1
        # cond1 = ((e1 == edge_index[0,:]) & (e2 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        # cond2 = ((e2 == edge_index[0,:]) & (e1 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        cond1 = ((e1 == edge_index[0, :]) & (e2 == edge_index[1, :])).nonzero(
            as_tuple=True
        )[0]
        cond2 = ((e2 == edge_index[0, :]) & (e1 == edge_index[1, :])).nonzero(
            as_tuple=True
        )[0]
        # print(cond1)

        if cond1.shape[0] > 0:
            edges_map[(e1, e2)] = cond1[0].item()
            edges_map[(e2, e1)] = cond1[0].item()
        elif cond2.shape[0] > 0:
            edges_map[(e1, e2)] = cond2[0].item()
            edges_map[(e2, e1)] = cond2[0].item()
        else:
            raise ValueError("Edge not in graph")

        # if cond1.shape[0] > 0 and cond2.shape[0] > 0:
        #     # Choose smallest
        #     edges_map[(e1, e2)] = min(cond1[0].item(), cond2[0].item())

        # # Check e1 -> 1, e2 -> 0 if the first condition didn't work
        # else:
        #     if cond1.shape[0] == 0:
        #         if cond2.shape[0] > 0:
        #             edges_map[(e2, e1)] = i
        #         else:
        #             #print(e1, e2)
        #             raise ValueError('Edge not in graph')
        #     else:
        #         edges_map[(e1, e2)] = i # Get first instance, don't care about duplicates

    return edges_map
