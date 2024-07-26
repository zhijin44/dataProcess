import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from powerMap2Graph import GraphDataset
from tqdm import tqdm
import optuna
import logging


logging.basicConfig(filename="optuna_GCN_per.log",
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, activation):
        super(GCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = activation

        # Input layer
        self.convs.append(GCNConv(num_node_features, hidden_channels[0]))

        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(GCNConv(hidden_channels[i - 1], hidden_channels[i]))

        # Output layer
        self.fc = torch.nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)

        # Use global mean pooling to get graph-level embeddings
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        return x


def print_best_params(study, trial):
    # Log the params of the current trial
    logging.info(f"Trial {trial.number} params: {trial.params}")

    best_trial = study.best_trial
    print("Best trial so far:")
    print(f"  Value: {best_trial.value}")
    logging.info(f"Best value: {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        logging.info(f"Best params: {key} = {value}")

    print(f"\nCompleted trials {trial.number}")
    logging.info(f"Completed trials {trial.number}")


# Load the dataset
dataset = torch.load('dataset/dataset_per.pt')
# dataset = torch.load('dataset_con.pt')

# Split the dataset into train and test sets (RANDOMLY)
torch.manual_seed(42)
train_len = int(len(dataset) * 0.9)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_channels = [trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32, 64, 128, 256]) for i in
                       range(num_layers)]
    activation_name = trial.suggest_categorical("activation", ["relu", "leaky_relu", "sigmoid", "tanh"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp", "Adagrad"])

    if activation_name == "relu":
        activation = F.relu
    elif activation_name == "leaky_relu":
        activation = F.leaky_relu
    elif activation_name == "sigmoid":
        activation = torch.sigmoid
    elif activation_name == "tanh":
        activation = torch.tanh

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and optimizer
    model = GCNModel(num_node_features=2, hidden_channels=hidden_channels, num_classes=1, activation=activation)
    model = model.double().to(device)  # Convert model parameters to double precision and move to device
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    accuracy_rate = 0.5

    # Train the model
    model.train()
    for epoch in tqdm(range(301)):
        train_loss, correct_predictions, total_predictions = 0, 0, 0
        for data in train_loader:
            data = data.to(device)  # move data to device
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            # loss = weighted_mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_predictions += torch.sum(torch.abs(out - data.y) / torch.abs(data.y) <= accuracy_rate).item()
            total_predictions += data.y.size(0)
        train_loss /= len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        # Test the model after each epoch
        model.eval()
        with torch.no_grad():
            test_loss, correct_predictions, total_predictions = 0, 0, 0
            for data in test_loader:
                data = data.to(device)  # move data to device
                out = model(data)
                test_loss += F.mse_loss(out, data.y).item()
                # test_loss += weighted_mse_loss(out, data.y).item()
                correct_predictions += torch.sum(torch.abs(out - data.y) / torch.abs(data.y) <= accuracy_rate).item()
                total_predictions += data.y.size(0)
            test_loss /= len(test_loader)
            test_accuracy = correct_predictions / total_predictions

        model.train()  # Switch model back to training mode
        scheduler.step()  # Step the learning rate scheduler

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, "
                f"Test Accuracy: {test_accuracy}")

    return test_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[print_best_params])
