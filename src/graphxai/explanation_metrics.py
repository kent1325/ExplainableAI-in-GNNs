import numpy as np


def fidelity(masked_graphs):
    fidelity_plus = 0
    fidelity_minus = 0
    for i, mg in enumerate(masked_graphs):
        for j, graph in enumerate(mg):
            if i == 0:
                fidelity_plus += np.sum(graph.y_pred == graph.y.item()) - np.sum(
                    graph.y_masked_pred == graph.y.item()
                )
            else:
                fidelity_minus += np.sum(graph.y_pred == graph.y.item()) - np.sum(
                    graph.y_masked_pred == graph.y.item()
                )

    fidelity_plus = fidelity_plus / len(masked_graphs[0])
    fidelity_minus = fidelity_minus / len(masked_graphs[1])

    return fidelity_plus, fidelity_minus


def sparsity(important_masked_graphs):
    sparsity = 0
    for img in important_masked_graphs:
        # num_of_imp_nodes = np.sum(np.all(img.x == 0, axis=1))
        num_of_imp_nodes = sum(
            all(element == 0 for element in sublist) for sublist in img.x
        )
        num_of_nodes = len(img.x)
        sparsity += 1 - (num_of_imp_nodes / num_of_nodes)
    sparsity = sparsity / len(important_masked_graphs)

    return sparsity


def contrastivity(masked_graphs):
    raise NotImplementedError
