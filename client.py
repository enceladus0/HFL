import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define the model (same as the server)
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

# Define the client
class Client(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=2)

    # ...the rest of the Client class code...

if __name__ == "__main__":
    client = Client(get_model())
    fl.client.start_numpy_client("192.168.126.93:5000", client)  # Use the actual IP of the edge server

