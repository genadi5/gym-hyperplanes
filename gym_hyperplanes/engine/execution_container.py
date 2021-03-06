class ExecutionContainer:
    def __init__(self, deep_level, config, data_provider, boundaries):
        # level of area search
        self.deep_level = deep_level
        self.config = config
        self.data_provider = data_provider
        # external boundaries - these we got from an area of previous level
        self.boundaries = boundaries

    def get_deep_level(self):
        return self.deep_level

    def get_config(self):
        return self.config

    def get_data_provider(self):
        return self.data_provider

    def get_boundaries(self):
        return self.boundaries

    def get_data_size(self):
        return self.data_provider.get_actual_data_size()
