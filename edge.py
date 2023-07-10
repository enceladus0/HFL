import flwr as fl
import torch
import torch.nn as nn

# Define the model (same as the client)
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

class EdgeServer(fl.server.Server):
    def __init__(self):
        self.model = get_model()

    def configure(self, rounds, min_clients, max_clients):
        return fl.server.strategy.FedAvg(rounds=rounds, min_clients=min_clients, max_clients=max_clients)

if __name__ == "__main__":
    edge = EdgeServer()
    strategy = edge.configure(10, 2, 2)
    fl.server.start_server("192.168.126.93:5000", strategy=strategy)  # Use the actual IP of the edge server

    client = Client(get_model())
    fl.client.start_numpy_client("192.168.126.61:5000", client)  # Use the actual IP of the cloud server

