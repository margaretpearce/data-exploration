class Interactions(object):
    def __init__(self, name, features=None, feature_interactions=None):
        # Data set name
        self.name = name

        # Feature names and IDs
        self.features = features

        # Feature-to-feature comparisons
        self.feature_interactions = feature_interactions

