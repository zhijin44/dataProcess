import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from powerMap2Graph import GraphDataset


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        # Use global mean pooling to get graph-level embeddings
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        return x


# Load the dataset
dataset = torch.load('stacked_features_dataset.pt')

# Split the dataset into train and test sets (RANDOMLY)
torch.manual_seed(42)
train_len = int(len(dataset) * 0.9)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and optimizer
model = GCNModel(num_node_features=2, num_classes=2)
model = model.double().to(device)  # Convert model parameters to double precision and move to device
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
accuracy_rate = 0.05

# Train the model
model.train()
for epoch in range(1):  # 300 epochs
    total_loss, correct_predictions, total_predictions = 0, 0, 0
    for data in train_loader:
        data = data.to(device)  # move data to device
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_predictions += torch.sum(torch.abs(out - data.y) / torch.abs(data.y) <= accuracy_rate).item()
        total_predictions += data.y.size(0)
    train_accuracy = correct_predictions / total_predictions
    print(f"Epoch: {epoch + 1}, Train Loss: {total_loss / len(train_loader)}, Train Accuracy: {train_accuracy}")

    # Test the model after each epoch
    model.eval()
    with torch.no_grad():
        test_loss, correct_predictions, total_predictions = 0, 0, 0
        for data in test_loader:
            data = data.to(device)  # move data to device
            out = model(data)
            test_loss += F.mse_loss(out, data.y).item()
            correct_predictions += torch.sum(torch.abs(out - data.y) / torch.abs(data.y) <= accuracy_rate).item()
            total_predictions += data.y.size(0)
        test_accuracy = correct_predictions / total_predictions
        print(f"Epoch: {epoch + 1}, Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}")

    model.train()
    scheduler.step()  # Step the learning rate scheduler
