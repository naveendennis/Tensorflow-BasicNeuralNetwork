import os.path


def get_module(file_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location('src',
                                                  os.path.join(
                                                      os.path.dirname(os.path.realpath(__name__)),
                                                      'src',
                                                      file_name))
    src_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src_module)
    return src_module


class BasicNeuralNetwork:
    def __init__(self, feature_vector, label_vector):
        self.feature_vector = feature_vector
        self.label_vector = label_vector

    def train_neural_network(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    properties = get_module('Utils.py').Utils.initialize()
    header_names = properties["header_names"]
    obj = get_module('DataCook.py').DataCook(
        file_name=os.path.join(os.path.dirname(os.path.realpath(__name__)),
                               properties["data_location"],
                               properties["filename"]),
        features=header_names,
        labels=header_names[2:],
        delimiter='\t'
    )
