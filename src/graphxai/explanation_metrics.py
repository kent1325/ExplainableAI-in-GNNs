import numpy as np
import torch
from torch.nn import functional as F

from settings.config import DEVICE


def fidelity(masked_graphs):
    fidelity_plus = []
    fidelity_minus = []
    for i, mg in enumerate(masked_graphs):
        for j, graph in enumerate(mg):
            if i == 0:
                fidelity_plus.append(
                    (graph.y_pred.item() == graph.y.item()) - (graph.y_masked_pred.item() == graph.y.item())
                )
            else:
                fidelity_minus.append(
                    (graph.y_pred.item() == graph.y.item()) - (graph.y_masked_pred.item() == graph.y.item())
                )

    return np.mean(fidelity_plus), np.mean(fidelity_minus)


def sparsity(important_masked_graphs):
    sparsity = []
    for img in important_masked_graphs:
        # num_of_imp_nodes = np.sum(np.all(img.x == 0, axis=1))
        num_of_imp_nodes = sum(
            all(element == 0 for element in sublist) for sublist in img.x
        )
        num_of_nodes = len(img.x)
        sparsity.append(1 - (num_of_imp_nodes / num_of_nodes))

    return np.mean(sparsity), np.std(sparsity)


def contrastivity(model, test_dataset, threshold=0.1):
    hamming_distance = []
    for i, graph in enumerate(test_dataset):
        model.eval()
        with torch.no_grad():
            prediction = model(
                graph.x,
                graph.edge_index,
                batch_index=torch.zeros(
                    1,
                    dtype=torch.int64,
                    device=DEVICE,
                ),
            )
        winner_class = prediction.max(dim=1)[1].item()
        winner_node_importance = torch.matmul(
            model.final_conv_acts,
            model.output.weight[winner_class],
        )
        winner_node_importance = F.relu(winner_node_importance)
        winner_norm_imp = (winner_node_importance) / (
            torch.max(winner_node_importance) + 1e-16
        )

        loser_class = winner_class ^ 1
        loser_node_importance = torch.matmul(
            model.final_conv_acts,
            model.output.weight[loser_class],
        )
        loser_node_importance = F.relu(loser_node_importance)
        loser_norm_imp = (loser_node_importance) / (torch.max(winner_node_importance) + 1e-16)

        winner_masking = [
            1 if v > threshold else 0 for i, v in enumerate(winner_norm_imp)
        ]
        loser_masking = [
            1 if v > threshold else 0 for i, v in enumerate(loser_norm_imp)
        ]
        ham_dist = sum(a != b for a, b in zip(winner_masking, loser_masking)) / len(
            winner_masking
        )

        hamming_distance.append(ham_dist)

    return np.mean(hamming_distance), np.std(hamming_distance)
