class Feature(object):
    def __init__(self, feat_name, feat_type, feat_count, feat_missing, feat_unique=None, feat_average=None,
                 feat_median=None, feat_max=None, feat_min=None, feat_stddev=None, feat_variance=None,
                 feat_mostcommon=None, feat_leastcommon=None, graph_hist=None):
        # Feature stats
        self.feat_name = feat_name
        self.feat_type = feat_type
        self.feat_count = feat_count
        self.feat_missing = feat_missing
        self.feat_unique = feat_unique

        # Numeric only
        self.feat_average = feat_average
        self.feat_median = feat_median
        self.feat_max = feat_max
        self.feat_min = feat_min
        self.feat_stddev = feat_stddev
        self.feat_variance = feat_variance

        # Categorical only
        self.feat_mostcommon = feat_mostcommon
        self.feat_leastcommon = feat_leastcommon

        # Visualizations
        self.graph_hist = graph_hist