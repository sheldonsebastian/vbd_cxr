import yaml


class Flags:

    def __init__(self):
        self.configs = {}

    def update(self, param_dict):
        # Add/Update new values to flags dictionary
        for key, value in param_dict.items():
            self.configs[key] = value
        return self

    def get(self, key):
        return self.configs[key]

    def get_configs(self):
        return self.configs

    def __repr__(self):
        return str(self.configs)

    def save_yaml(self, filepath):
        with open(filepath, "w") as f:
            yaml.dump(self.configs, f)

    def load_yaml(self, filepath):
        with open(filepath, "r") as f:
            self.configs = yaml.full_load(f)
        return self
