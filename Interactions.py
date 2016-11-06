class Interactions(object):
    def __init__(self, name, correlations_url=None, covariance_url=None):
        self.name = name
        self.correlations_url = correlations_url
        self.covariance_url = covariance_url