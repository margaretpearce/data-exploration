import os
import jsonpickle
import seaborn as sns

from configuration import const_types
from configuration import paths
from controllers.data_driver import DataDriver
from model.feature import Feature
from model.features import Features


class DataUnivariate(DataDriver):
    def __init__(self, selected_dataset):
        DataDriver.__init__(self, selected_dataset)

    def load_features_json(self):
        features_json = self.load_json(paths.FEATURES_SUFFIX)

        # If the file doesn't exist, generate it
        if features_json is None:
            self.generate_features_json()
            features_json = self.load_json(paths.FEATURES_SUFFIX)

        # Return the JSON
        return features_json

    def generate_features_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:
            features_collection = []
            feature_index = 0

            for feat_name in self.data.columns.values:
                feature = self.get_feature(feat_name, feature_index)
                features_collection.append(feature)
                feature_index += 1

            # Create object holding features collection and save as JSON
            features = Features(self.title, features_collection)
            features_json = jsonpickle.encode(features)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=features_json, suffix=paths.FEATURES_SUFFIX)

    def get_feature(self, feat_name, feature_index):
        var_datatype = self.get_data_type(feat_name)
        var_vartype = self.get_vartype_formatted(feat_name)
        var_count = self.get_count(feat_name)
        var_missing = self.get_missing_formatted(feat_name)
        var_unique = self.get_count_unique(feat_name)

        # Numeric only
        var_avg = self.format_rounded_string(self.get_average(feat_name))
        var_median = self.get_median(feat_name)
        var_mode = self.get_mode(feat_name)
        var_max = self.get_max(feat_name)
        var_min = self.get_min(feat_name)
        var_stddev = self.format_rounded_string(self.get_stddev(feat_name))
        var_variance = self.format_rounded_string(self.get_variance(feat_name))
        var_quantile25 = self.format_rounded_string(self.get_quantile25(feat_name))
        var_quantile75 = self.format_rounded_string(self.get_quantile75(feat_name))
        var_iqr = self.format_rounded_string(self.get_iqr(feat_name))
        var_skew = self.format_rounded_string(self.get_skew(feat_name))
        var_kurtosis = self.format_rounded_string(self.get_kurtosis(feat_name))

        # Non-numeric only
        var_mostcommon = self.get_mostcommon(feat_name)
        var_leastcommon = self.get_leastcommon(feat_name)

        # Graphs
        graph_histogram = self.get_histogram(feat_name)
        graph_countplot = self.get_countplot(feat_name)

        # Errors, warnings, and info
        feat_errors = self.get_errors(feat_name)
        feat_warnings = self.get_warnings(feat_name)
        feat_notes = self.get_notes(feat_name)

        # Save the feature stats
        feature = Feature(feat_name=feat_name,
                          feat_index=feature_index,
                          feat_datatype=var_datatype,
                          feat_vartype=var_vartype,
                          feat_count=var_count,
                          feat_missing=var_missing,
                          feat_unique=var_unique,
                          feat_average=var_avg,
                          feat_median=var_median,
                          feat_mode=var_mode,
                          feat_max=var_max,
                          feat_min=var_min,
                          feat_stddev=var_stddev,
                          feat_variance=var_variance,
                          feat_quantile25=var_quantile25,
                          feat_quantile75=var_quantile75,
                          feat_iqr=var_iqr,
                          feat_skew=var_skew,
                          feat_kurtosis=var_kurtosis,
                          feat_mostcommon=var_mostcommon,
                          feat_leastcommon=var_leastcommon,
                          graph_histogram=graph_histogram,
                          graph_countplot=graph_countplot,
                          feat_errors=feat_errors,
                          feat_warnings=feat_warnings,
                          feat_notes=feat_notes)
        return feature

    def get_count(self, feat_name):
        return int(self.data[feat_name].count())

    def get_count_missing(self, feat_name):
        return int(self.data[feat_name].isnull().sum())

    def get_percent_missing(self, feat_name):
        missing_count = self.get_count_missing(feat_name)
        missing_percent = 100 * missing_count / float(self.data.shape[0])
        return missing_percent

    def get_missing_formatted(self, feat_name):
        return str("%s (%.3f%%)" % (self.get_count_missing(feat_name), self.get_percent_missing(feat_name)))

    def get_vartype_formatted(self, feat_name):
        vartype = self.get_variable_type(feat_name)

        # Denote label and index, if applicable
        if self.id_column is not None and feat_name == self.id_column:
            vartype += " (ID)"
        elif self.label_column is not None and feat_name == self.label_column:
            vartype += " (Label)"

        return vartype

    def feat_is_numeric(self, feat_name):
        return self.data[feat_name].dtype in ['int64', 'float64']

    def get_average(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return float(self.data[feat_name].mean())
        return None

    def get_median(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return float(self.data[feat_name].median())
        return None

    def get_mode(self, feat_name):
        var_mode = ""
        mode = self.data[feat_name].mode()

        if mode is not None:
            for m in mode:
                var_mode = var_mode + str(m) + " "

        # If no mode is found, return None instead of empty string
        if var_mode == "":
            var_mode = None

        return var_mode

    def get_max(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return float(self.data[feat_name].max())
        return None

    def get_min(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return float(self.data[feat_name].min())
        return None

    def get_stddev(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].std()
        return None

    def get_variance(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].var()
        return None

    def get_quantile25(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].dropna().quantile(q=0.25)
        return None

    def get_quantile75(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].dropna().quantile(q=0.75)
        return None

    def get_iqr(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.get_quantile75(feat_name) - self.get_quantile25(feat_name)
        return None

    def get_skew(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].skew()
        return None

    def get_kurtosis(self, feat_name):
        if self.feat_is_numeric(feat_name):
            return self.data[feat_name].kurt()
        return None

    def get_mostcommon(self, feat_name):
        if not self.feat_is_numeric(feat_name):
            return str("%s (%d)" % (self.data[feat_name].value_counts().idxmax(),
                                    self.data[feat_name].value_counts().max()))
        return None

    def get_leastcommon(self, feat_name):
        if not self.feat_is_numeric(feat_name):
            return str("%s (%d)" % (self.data[feat_name].value_counts().idxmin(),
                                    self.data[feat_name].value_counts().min()))
        return None

    def get_histogram(self, feat_name):
        if self.get_data_type(feat_name) in [const_types.DATATYPE_FLOAT, const_types.DATATYPE_INTEGER]:
            hist_plot = sns.distplot(self.data[feat_name].dropna(), bins=None, hist=True, kde=False, rug=False)
            return self.save_graph(hist_plot, feat_name + paths.FILE_HISTOGRAM)
        return None

    def get_countplot(self, feat_name):
        if self.get_data_type(feat_name) not in [const_types.DATATYPE_FLOAT, const_types.DATATYPE_INTEGER] and \
           self.check_uniques_for_graphing(feat_name):
                countplot = sns.countplot(y=self.data[feat_name].dropna())
                return self.save_graph(countplot, filename=feat_name + paths.FILE_COUNTPLOT)
        return None

    def get_errors(self, feat_name):
        return None

    def get_warnings(self, feat_name):
        warnings = []
        if self.get_percent_unique(feat_name) == 1:
            warnings.append("This feature has all unique values")
        if self.get_percent_missing(feat_name) >= 50:
            warnings.append("This feature is missing in 50% or more rows")
        return warnings

    def get_notes(self, feat_name):
        notes = []
        if self.get_percent_missing(feat_name) == 0:
            notes.append("This feature is not missing any values")
        return notes
