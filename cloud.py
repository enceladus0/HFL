import flwr as fl

class Cloud:
    def configure_server(self):
        """Returns a FedAvg strategy for the server"""
        return fl.server.strategy.FedAvg(
            fraction_fit=0.5,  # 50% of clients used for training in each round
            min_fit_clients=2,  # minimum number of clients to be used for training
            min_available_clients=2,  # minimum number of clients that need to be connected to the server
        )

if __name__ == "__main__":
    cloud = Cloud()
    strategy = cloud.configure_server()
    fl.server.start_server("0.0.0.0:5000", strategy=strategy, num_rounds=3)

