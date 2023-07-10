import flwr as fl
import torch
import torch.nn as nn

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

class Cloud(fl.server.Server):
    def __init__(self):
        self.model = get_model()

    def configure(self, min_clients, max_clients):
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=min_clients,
            min_eval_clients=max_clients
        )

if __name__ == "__main__":
    cloud = Cloud()
    strategy = cloud.configure(2, 2)
    fl.server.start_server("0.0.0.0:5000", config=strategy)
