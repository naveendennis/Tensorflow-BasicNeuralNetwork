import json
import os


class Utils:
    @staticmethod
    def initialize():
        with open(os.path.join(os.path.realpath(__name__),
                               'properties.json'), 'r') as f:
            return json.load(f)
