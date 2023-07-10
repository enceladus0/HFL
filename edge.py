import flwr as fl
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flwr.common import Weights, parameters_to_weights, weights_to_parameters

def get_model():
    model = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return parameters_to_weights(self.model.parameters())

    def set_parameters(self, parameters):
        self.model.parameters = weights_to_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(2):  # 2 epochs
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return loss, len(self.test_loader), {"accuracy": correct / total}

class EdgeServer(fl.server.Server):
    def __init__(self, model):
        self.model = model

    def configure(self, min_clients, max_clients):
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=min_clients,
            min_eval_clients=max_clients
        )

if __name__ == "__main__":
    model = get_model()
    edge = EdgeServer(model)
    strategy = edge.configure(2, 2)
    fl.server.start_server("192.168.126.93:5000", strategy=strategy)

    client = EdgeClient(model)
    fl.client.start_numpy_client("192.168.126.49:5000", client)
