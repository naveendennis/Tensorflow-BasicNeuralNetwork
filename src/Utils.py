import json
import os


class Utils:
    @staticmethod
    def initialize():
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'properties.json'), 'r') as f:
            return json.load(f)
