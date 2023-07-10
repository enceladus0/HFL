import flwr as fl
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# same model as before
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

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    def get_parameters(self):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(w).to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for epoch in range(2):  # 2 epochs
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 391 == 390:    # print every 390 mini-batches, 781 mini-batches total
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
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
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        return loss, len(self.test_loader), {"accuracy": correct / total}

if __name__ == "__main__":
    client = Client()
    fl.client.start_numpy_client("192.168.126.93:5000", client)
