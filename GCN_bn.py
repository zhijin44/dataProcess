import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from tqdm import tqdm
import optuna
import logging

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename="log/GCN_doubleData_1000.log",
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


class GraphDataset(Dataset):
    def __init__(self, root, features, edge_index, labels, transform=None, pre_transform=None):
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    def len(self):
        return len(self.features)

    def get(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data


def weighted_mse_loss(out, labels):
    # Separate the output and labels into conductivity and permittivity components
    out_conductivity, out_permittivity = out[:, 0], out[:, 1]
    labels_conductivity, labels_permittivity = labels[:, 0], labels[:, 1]

    # Compute the MSE loss for each task
    loss_conductivity = F.mse_loss(out_conductivity, labels_conductivity)
    loss_permittivity = F.mse_loss(out_permittivity, labels_permittivity)

    # Dynamically adjust the weight based on current loss
    weight_conductivity = loss_permittivity / (loss_conductivity + 1e-8)

    # Compute the combined loss
    loss = weight_conductivity * loss_conductivity + loss_permittivity

    return loss


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()  # Create a new ModuleList for batch normalization layers

        # Input layer
        self.convs.append(SAGEConv(num_node_features, hidden_channels[0]))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels[0]))

        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(SAGEConv(hidden_channels[i - 1], hidden_channels[i]))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels[i]))

        # Output layer
        self.fc = torch.nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)  # Apply batch normalization
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # Use global mean pooling to get graph-level embeddings
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        return x


# Load the dataset
dataset = torch.load('dataset/dataset.pt')

# Split the dataset into train and test sets (RANDOMLY)
torch.manual_seed(42)
train_len = int(len(dataset) * 0.9)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)

# Initialize the model and optimizer
model = GCNModel(num_node_features=2, hidden_channels=[256, 128, 32], num_classes=2)
model = model.double().to(device)  # Convert model parameters to double precision and move to device
optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
accuracy_rate = 0.07

# Train the model
model.train()
for epoch in tqdm(range(401)):
    train_loss, correct_predictions, total_predictions = 0, 0, 0
    for data in train_loader:
        data = data.to(device)  # move data to device
        optimizer.zero_grad()
        out = model(data)
        # loss = F.mse_loss(out, data.y)
        loss = weighted_mse_loss(out, data.y)
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
            # test_loss += F.mse_loss(out, data.y).item()
            test_loss += weighted_mse_loss(out, data.y).item()
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
        logging.info(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, "
            f"Test Accuracy: {test_accuracy}")
