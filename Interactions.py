class Interactions(object):
    def __init__(self, name, correlations_url=None, covariance_url=None, features=[], feature_interactions={}):
        # Data set name
        self.name = name

        # Feature names and IDs
        self.features = features

        # Feature-to-feature comparisons
        self.feature_interactions = feature_interactions

        self.correlations_url = correlations_url
        self.covariance_url = covariance_url