import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from powerMap2Graph import GraphDataset
from tqdm import tqdm


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
dataset = torch.load('dataset/dataset_per.pt')

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
model = GCNModel(num_node_features=2, num_classes=1)
model.double()  # Convert model parameters to double precision
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
accuracy_rate = 0.05

# Train the model
model.train()
for epoch in tqdm(range(1)):  # Add progress bar
    train_loss, correct_predictions, total_predictions = 0, 0, 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct_predictions += torch.sum(torch.abs(out - data.y) / torch.abs(data.y) <= accuracy_rate).item()
        total_predictions += data.y.size(0)
    train_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    # Test the model after each epoch
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        test_loss, correct_predictions, total_predictions = 0, 0, 0
        for data in test_loader:
            out = model(data)
            test_loss += F.mse_loss(out, data.y).item()
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

