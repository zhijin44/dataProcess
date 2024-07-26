import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Dataset, Batch
from models import GCNModel, SAGEModel, GINModel
from tqdm import tqdm
import random
import optuna
import logging

# logging settings
logging.basicConfig(filename="log/optuna_GIN_per.log",
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# 在所有代码执行之前设置种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def prep_data(dataset, batch_size):
    # Load the dataset
    dataset = torch.load(dataset)

    # Split the dataset into train and test sets (RANDOMLY)
    torch.manual_seed(42)
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # Create data loaders
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)

    return train_loader, test_loader


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, out, labels):
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


def approximate(data):
    mask = (data > 0.5) & (data < 109.5)
    data_rounded = data.clone()
    data_rounded[mask] = torch.round(data[mask])
    return data_rounded


def map_to_class(data):
    # 初始化一个空的tensor用于存放结果
    data_class = torch.empty_like(data)

    # 对数据中的0进行处理，将其映射到118类
    zero_class = (data == 0)
    data_class[zero_class] = 118

    # 对非零的数据进行映射
    non_zero_data = data[~zero_class]
    magnitude = torch.floor(torch.log10(torch.abs(non_zero_data)))
    data_class[~zero_class] = (-(magnitude + 3) * 9) + torch.floor(non_zero_data * 10 ** -magnitude)

    return data_class


def map_to_value(data_class):
    # Handle the special case of 0
    original_data = torch.zeros_like(data_class, dtype=torch.float32)
    non_zero_class_mask = data_class != 118

    # Calculate magnitude and value from class
    magnitude = torch.div(data_class[non_zero_class_mask].float() - 1, 9, rounding_mode='floor') * -1 - 3
    value = ((data_class[non_zero_class_mask].float() - 1) % 9 + 1)

    # Recover original data
    original_data[non_zero_class_mask] = value * (10 ** magnitude)

    return original_data


def accuracy(pred_out, label, accuracy_rate):
    correct_predictions = torch.sum(torch.abs(pred_out - label) / torch.abs(label) <= accuracy_rate).item()
    batch_size = label.size(0)
    # print(batch_size)
    return correct_predictions / batch_size


@torch.no_grad()
def test(model, loader, criterion, accuracy_rate):
    model.eval()
    test_loss, test_acc = 0, 0
    for data in loader:
        data = data.to(device)  # move data to device
        out = model(data)
        ########## 对 data.y 进行映射, test for conductivity ############
        # mapped_y = map_to_class(data.y) #
        # test_loss += criterion(out, mapped_y).item() / len(loader)
        # appr_out = approximate(out)
        # test_acc +=  accuracy(appr_out, mapped_y, accuracy_rate) / len(loader)
        ################################################################
        test_loss += criterion(out, data.y).item() / len(loader)
        test_acc += accuracy(out, data.y, accuracy_rate) / len(loader)
    return test_loss, test_acc


def train(model, optimizer, criterion, batch_size, epochs, dataset, rate):
    # load data
    train_loader, test_loader = prep_data(dataset, batch_size)

    # Initialize the model and optimizer
    model = model.double().to(device)  # Convert model parameters to double precision and move to device
    optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = criterion
    accuracy_rate = rate
    epochs = epochs

    # Train the model
    model.train()
    for epoch in tqdm(range(epochs)):  # Add progress bar
        train_loss, train_accuracy = 0, 0
        for data in train_loader:
            data = data.to(device)  # move data to device
            optimizer.zero_grad()
            out = model(data)
            ###### 对 data.y 进行映射, train for conductivity ######
            # mapped_y = map_to_class(data.y)
            # loss = criterion(out, mapped_y)
            # train_loss += loss.item() / len(train_loader)
            # appr_out = approximate(out)
            # train_accuracy += accuracy(appr_out, mapped_y, accuracy_rate) / len(train_loader)
            ###########################################################
            loss = criterion(out, data.y)
            train_loss += loss.item() / len(train_loader)
            train_accuracy += accuracy(out, data.y, accuracy_rate) / len(train_loader)
            loss.backward()
            optimizer.step()

        # Test the model after each epoch
        test_loss, test_accuracy = test(model=model, loader=test_loader, criterion=criterion,
                                        accuracy_rate=accuracy_rate)

        scheduler.step()  # Step the learning rate scheduler
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy * 100:.2f}%, | Test Loss: {test_loss:.2f}, "
                f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return test_accuracy


class Objective:
    def __init__(self, criterion, epochs, dataset, rate):
        self.criterion = criterion
        self.epochs = epochs
        self.dataset = dataset
        self.rate = rate

    def __call__(self, trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256])
        conv_layers = trial.suggest_categorical('conv_layers', [3, 4, 5])

        current_model = GINModel(num_node_features=2, num_classes=1, dim_h=hidden_dim, layers=conv_layers)
        optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)

        test_accuracy = train(model=current_model, optimizer=optimizer, criterion=self.criterion, batch_size=batch_size,
                              epochs=self.epochs, dataset=self.dataset, rate=self.rate)

        return test_accuracy


def print_best_params(study, trial):
    print(f"Start trial {trial.number} params: {trial.params}")
    logging.info(f"Start trial {trial.number} params: {trial.params}")

    best_trial = study.best_trial
    print("Best trial so far:")
    print(f"  Value: {best_trial.value * 100:.2f}%")
    logging.info(f"Best value: {best_trial.value * 100:.2f}%")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        logging.info(f"Best params: {key} = {value}")

    print(f"\nCompleted trials {trial.number}")
    logging.info(f"Completed trials {trial.number}")


if __name__ == '__main__':
    ################ for single trail training ###############################
    # Model = GINModel(num_node_features=2, num_classes=1, dim_h=64)
    # Optimizer = torch.optim.Adam(Model.parameters(), lr=0.0005)
    # # Criterion = WeightedMSELoss()
    # Criterion = torch.nn.MSELoss()
    # Batch_size = 32
    # Epochs = 101
    # Dataset = './dataset/dataset_per.pt'
    # Rate = 0.5
    #
    # train(model=Model, optimizer=Optimizer, criterion=Criterion, batch_size=Batch_size, epochs=Epochs, dataset=Dataset, rate=Rate)
    #############################################################################
    Model = GINModel(num_node_features=2, num_classes=1, dim_h=256, layers=6)
    Optimizer = torch.optim.Adam(Model.parameters(), lr=0.000164)
    Criterion = torch.nn.MSELoss()
    Batch_size = 16
    Epochs = 11
    Dataset = './dataset/dataset_per.pt'
    Rate = 0.07

    train(model=Model, optimizer=Optimizer, criterion=Criterion, batch_size=Batch_size, epochs=Epochs, dataset=Dataset, rate=Rate)
    ##########################################################################

    ################## tune super-parameter by optuna ########################
    # objective = Objective(
    #     criterion=torch.nn.MSELoss(),
    #     epochs=251,
    #     dataset='./dataset/dataset_per.pt',
    #     rate=0.07
    # )
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50, callbacks=[print_best_params])
    ##########################################################################
