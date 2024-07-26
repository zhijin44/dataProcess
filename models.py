import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool
from powerMap2Graph import GraphDataset


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, activation, dropout_rate):
        super(GCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = activation
        self.dropout_rate = dropout_rate

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
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Use global mean pooling to get graph-level embeddings
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        return x


class SAGEModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, activation, dropout_rate):
        super(SAGEModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Input layer
        self.convs.append(SAGEConv(num_node_features, hidden_channels[0]))

        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(SAGEConv(hidden_channels[i - 1], hidden_channels[i]))

        # Output layer
        self.fc = torch.nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Use global mean pooling to get graph-level embeddings
        x = global_add_pool(x, data.batch)

        x = self.fc(x)
        return x


class GINModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, dim_h, layers):
        super(GINModel, self).__init__()

        self.layers = layers

        # 初始化多个卷积层
        self.convs = torch.nn.ModuleList()
        for _ in range(self.layers):
            self.convs.append(
                GINConv(
                    Sequential(Linear(num_node_features if _ == 0 else dim_h, dim_h),
                               BatchNorm1d(dim_h), ReLU(),
                               Linear(dim_h, dim_h), ReLU())
                )
            )

        self.lin1 = Linear(dim_h * layers, dim_h * layers)
        self.lin2 = Linear(dim_h * layers, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embeddings
        node_embeddings = []
        for i in range(self.layers):
            if i == 0:
                node_embeddings.append(self.convs[i](x, edge_index))
            else:
                node_embeddings.append(self.convs[i](node_embeddings[-1], edge_index))

        # Graph-level readout for each layer's embedding
        graph_embeddings = [global_add_pool(h, batch) for h in node_embeddings]

        # Concatenate graph embeddings
        h = torch.cat(graph_embeddings, dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

# class GINModel(torch.nn.Module):
#     def __init__(self, num_node_features, num_classes, dim_h):
#         super(GINModel, self).__init__()
#         self.conv1 = GINConv(
#             Sequential(Linear(num_node_features, dim_h),
#                        BatchNorm1d(dim_h), ReLU(),
#                        Linear(dim_h, dim_h), ReLU()))
#         self.conv2 = GINConv(
#             Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
#                        Linear(dim_h, dim_h), ReLU()))
#         self.conv3 = GINConv(
#             Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
#                        Linear(dim_h, dim_h), ReLU()))
#         self.lin1 = Linear(dim_h * 3, dim_h * 3)
#         self.lin2 = Linear(dim_h * 3, num_classes)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         # Node embeddings
#         h1 = self.conv1(x, edge_index)
#         h2 = self.conv2(h1, edge_index)
#         h3 = self.conv3(h2, edge_index)
#
#         # Graph-level readout
#         h1 = global_add_pool(h1, batch)
#         h2 = global_add_pool(h2, batch)
#         h3 = global_add_pool(h3, batch)
#
#         # Concatenate graph embeddings
#         h = torch.cat((h1, h2, h3), dim=1)
#
#         # Classifier
#         h = self.lin1(h)
#         h = h.relu()
#         h = F.dropout(h, p=0.2, training=self.training)
#         h = self.lin2(h)
#
#         return h


