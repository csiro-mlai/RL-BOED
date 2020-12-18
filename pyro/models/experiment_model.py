from abc import ABC


class ExperimentModel(ABC):
    def __init__(self, network):
        self.network = network

    def copy(self):
        raise NotImplementedError

    def run_experiment(self, design):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
