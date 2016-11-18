class Interaction(object):
    def __init__(self, feat_name, feat_index, other_features=[None], scatterplots={}, correlations={}, covariances={},
                 boxplots={}, ztests={}, ttests={}, anova={}, stackedbarplots={}, chisquared={}, cramers={},
                 mantelhchi={}):
        # Feature comparing against all others in the data set
        self.feat_name = feat_name
        self.feat_index = feat_index
        self.other_features = other_features

        # Continuous & continuous
        self.scatterplots = scatterplots
        self.correlations = correlations
        self.covariances = covariances

        # Categorical & continuous
        self.boxplots = boxplots
        self.ztests = ztests
        self.ttests = ttests
        self.anova = anova

        # Categorical & categorical
        self.stackedbarplots = stackedbarplots
        self.chisquared = chisquared
        self.cramers = cramers
        self.mantelhchi = mantelhchi
