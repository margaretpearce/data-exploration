class Interaction(object):
    def __init__(self, feat_name, feat_index, other_features=[None], scatterplots={}, correlations={}, covariances={},
                 boxplots={}, statsbycategory={}, statsforcategory=[], ztests={}, ttests={}, anova={},
                 stackedbarplots={}, chisquared={}, cramers={}, mantelhchi={}, frequency_table={},
                 frequencytable_firstrow=None):
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
        self.statsbycategory = statsbycategory
        self.statsforcategory = statsforcategory
        self.ztests = ztests
        self.ttests = ttests
        self.anova = anova

        # Categorical & categorical
        self.stackedbarplots = stackedbarplots
        self.chisquared = chisquared
        self.cramers = cramers
        self.mantelhchi = mantelhchi
        self.frequency_table = frequency_table
        self.frequencytable_firstrow = frequencytable_firstrow

