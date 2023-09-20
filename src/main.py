from data.get_dataloader import MUTAGLoader
from networks.gnn_loader import GAT, GCN
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from torch_geometric.loader import DataLoader
from settings.config import GRAPH_BATCH_SIZE


if __name__ == "__main__":
    mutag_dataset = MUTAGLoader().get_dataset()
    model = GAT(mutag_dataset.num_features)
    print(model)
    print(f"Number of parameters: {model.count_parameters()}")

    model_trainer = ModelTrainer(model)
    model_tester = ModelTester(model)

    train_loader = DataLoader(
        mutag_dataset[: int(len(mutag_dataset) * 0.8)],
        batch_size=GRAPH_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        mutag_dataset[int(len(mutag_dataset) * 0.8) :],
        batch_size=GRAPH_BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(1, 101):
        model.train()
        train_loss, train_y_pred, train_y_true = model_trainer.train_model(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3f}")

            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
            print(f"Epoch {epoch} | Test Loss: {test_loss:.3f}")
