import os.path
from importlib import import_module


class BasicNeuralNetwork:
    def __init__(self, feature_vector, label_vector):
        self.feature_vector = feature_vector
        self.label_vector = label_vector

    def train_neural_network(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    properties = import_module('Utils').Utils.initialize()
    header_names = properties["header_names"]
    obj = import_module('DataCook').DataCook(
        file_name=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '..',
                               properties["data_location"],
                               properties["filename"]),
        features=header_names,
        labels=header_names[2:],
        delimiter='\t'
    )
