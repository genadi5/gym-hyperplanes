class ExecutionContainer:
    def __init__(self, deep_level, config, data_provider, external_hp_state, external_hp_dist, external_area):
        self.deep_level = deep_level
        self.config = config
        self.data_provider = data_provider
        self.external_hp_state = external_hp_state
        self.external_hp_dist = external_hp_dist
        self.external_area = external_area

    def get_deep_level(self):
        return self.deep_level

    def get_config(self):
        return self.config

    def get_data_provider(self):
        return self.data_provider

    def get_external_hp_state(self):
        return self.external_hp_state

    def get_external_hp_dist(self):
        return self.external_hp_dist

    def get_external_area(self):
        return self.external_area

    def get_data_size(self):
        return self.data_provider.get_data_size()
