import os
import re
from collections import OrderedDict

import jsonpickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from configuration import paths
from configuration import const_types
from model.feature import Feature
from model.features import Features
from model.interaction import Interaction
from model.interactions import Interactions
from model.summary import Summary


class DataDriver:
    def __init__(self, selected_dataset):
        self.file = selected_dataset[0]
        self.title = selected_dataset[1]
        self.id_column = selected_dataset[2]
        self.label_column = selected_dataset[3]
        self.file_uploaded = selected_dataset[4]

        if self.file_uploaded:
            self.filepath = os.path.join(paths.UPLOAD_FOLDER, self.title, str(self.file))
        else:
            self.filepath = os.path.join(paths.EXAMPLES_FOLDER, self.title, str(self.file))

        self.data = None
        self.error_code = None

    def load_data(self):
        # Load the data into a Pandas DataFrame
        try:
            if str(self.file).endswith("csv"):
                self.data = pd.read_csv(self.filepath)
            elif str(self.file).endswith("tsv"):
                self.data = pd.read_csv(self.filepath, sep='\t')
            elif str(self.file).endswith("xls") or str(self.file).endswith("xlsx"):
                self.data = pd.read_excel(self.filepath)
            return True

        except ValueError as err:
            self.error_code = str("{0}".format(err))
            return False

    def generate_summary_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:
            # Get summary stats about the data and serialize it as JSON
            num_records = self.data.shape[0]
            num_features = self.data.shape[1]
            index_column = self.id_column
            label_column = self.label_column

            # Count the number of columns missing for each row
            num_rows_missing = self.count_missing(num_records)

            # List of features
            features_list = list(self.data.columns.values)

            # Sample data (five rows or less)
            sample_list = self.get_sample(num_records, features_list)

            summary = Summary(name=self.title,
                              num_records=num_records,
                              num_features=num_features,
                              index_column=index_column,
                              label_column=label_column,
                              rows_missing=num_rows_missing,
                              features_list=features_list,
                              sample_list=sample_list
                              )
            summary_json = jsonpickle.encode(summary)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=summary_json, suffix=paths.SUMMARY_SUFFIX)

    def count_missing(self, num_records):
        # Count the number of columns missing for each row
        count_missing = self.data.apply(lambda x: sum(x.isnull().values), axis=1)
        self.data["num_missing"] = pd.Series(count_missing)

        num_rows_missing = {}
        cumulative_row_sum = 0

        for i in range(0, num_records):
            # Count the number of missing rows and add it to the list
            num_rows_i_missing = int(sum(self.data["num_missing"] == i))
            num_rows_missing[i] = num_rows_i_missing
            cumulative_row_sum += num_rows_i_missing

            # If we reached the point where we have accounted for all rows, stop looking for rows
            if cumulative_row_sum == num_records:
                break

        # Sort the dictionary
        num_rows_missing = OrderedDict(sorted(num_rows_missing.items()))

        # Remove the added column
        self.data.drop("num_missing", axis=1, inplace=True)

        return num_rows_missing

    def get_sample(self, num_records, features_list):
        num_samples = 5
        if num_records < num_samples:
            num_samples = num_records

        sample_list = self.data.sample(num_samples)[features_list].values.tolist()
        return sample_list

    def generate_features_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:

            features_collection = []
            feature_index = 0

            # Univariate analysis for each feature
            for var_name in self.data.columns.values:

                # Common for all field types
                var_datatype = self.get_data_type(var_name)
                var_vartype = self.get_variable_type(var_name)
                var_count = int(self.data[var_name].count())
                missing_count = int(self.data[var_name].isnull().sum())
                missing_percent = 100 * missing_count / float(self.data.shape[0])
                var_missing = str("%s (%.3f%%)" % (missing_count, missing_percent))
                var_unique = int(len(self.data[var_name].unique()))

                # Denote label and index, if applicable
                if self.id_column is not None and var_name == self.id_column:
                    var_vartype += " (ID)"
                elif self.label_column is not None and var_name == self.label_column:
                    var_vartype += " (Label)"

                # Numeric only
                var_avg = None
                var_median = None
                var_mode = None
                var_max = None
                var_min = None
                var_stddev = None
                var_variance = None
                var_quantile25 = None
                var_quantile75 = None
                var_iqr = None
                var_skew = None
                var_kurtosis = None

                # Non-numeric only
                var_mostcommon = None
                var_leastcommon = None

                # Graphs
                graph_histogram = None
                graph_countplot = None

                # Compute numeric statistics
                if self.data[var_name].dtype in ['int64', 'float64']:
                    var_avg = str("%.3f" % float(self.data[var_name].mean()))
                    var_median = float(self.data[var_name].median())
                    var_max = float(self.data[var_name].max())
                    var_min = float(self.data[var_name].min())
                    var_stddev = str("%.3f" % self.data[var_name].std())
                    var_variance = str("%.3f" % self.data[var_name].var())
                    var_quantile25 = str("%.3f" % self.data[var_name].dropna().quantile(q=0.25))
                    var_quantile75 = str("%.3f" % self.data[var_name].dropna().quantile(q=0.75))
                    var_iqr = str("%.3f" % (self.data[var_name].dropna().quantile(q=0.75) -
                                            self.data[var_name].dropna().quantile(q=0.25)))
                    var_skew = str("%.3f" % self.data[var_name].skew())
                    var_kurtosis = str("%.3f" % self.data[var_name].kurt())

                    var_mode = self.get_mode(var_name)

                # Compute non-numeric stats
                else:
                    var_mostcommon = str("%s (%d)" %
                                         (self.data[var_name].value_counts().idxmax(),
                                          self.data[var_name].value_counts().max()))
                    var_leastcommon = str("%s (%d)" %
                                          (self.data[var_name].value_counts().idxmin(),
                                           self.data[var_name].value_counts().min()))

                # Histogram (numeric)
                if self.get_data_type(var_name) in [const_types.DATATYPE_FLOAT, const_types.DATATYPE_INTEGER]:
                    hist_plot = sns.distplot(self.data[var_name].dropna(), bins=None, hist=True, kde=False, rug=False)
                    graph_histogram = self.save_graph(hist_plot, var_name + paths.FILE_HISTOGRAM)

                # Countplot (non-numeric)
                elif self.check_uniques_for_graphing(var_name):
                    countplot = sns.countplot(y=self.data[var_name].dropna())

                    if self.get_percent_unique(var_name) > 0.5:
                        countplot.set(ylabel='')
                        countplot.set(yticklabels=[])
                        countplot.yaxis.set_visible(False)

                    graph_countplot = self.save_graph(countplot, filename=var_name + paths.FILE_COUNTPLOT)

                # Save the feature stats
                feature = Feature(feat_name=var_name,
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
                                  graph_countplot=graph_countplot)
                features_collection.append(feature)
                feature_index += 1

            # Create object holding features collection and save as JSON
            features = Features(self.title, features_collection)
            features_json = jsonpickle.encode(features)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=features_json, suffix=paths.FEATURES_SUFFIX)

    def save_graph(self, plot, filename):
        folder_path = paths.EXAMPLES_FOLDER
        relative_path = paths.EXAMPLES_RELATIVE

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER
            relative_path = paths.UPLOAD_RELATIVE

        # Replace any special characters in the filename
        filename = re.sub(r'[^.a-zA-Z0-9_-]', '', filename)

        full_url = os.path.join(folder_path, self.title, str("graphs/" + filename))
        fig = plot.get_figure()
        fig.savefig(full_url)

        # Clear the figure to prepare for the next plot
        sns.plt.clf()

        # Return the relative URL to the histogram
        graph_url = relative_path + self.title + str(paths.GRAPHS_SUBFOLDER + filename)
        return graph_url

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

    def get_data_type(self, feat_name):
        raw_type = str(self.data[feat_name].dtype)
        var_datatype = None

        # Get the data type based on its raw type
        if raw_type == "int64" or raw_type == "int8":
            # Check if it's really a boolean
            unique_vals = self.data[feat_name].unique()
            for val in unique_vals:
                if not (int(val) == 0 or int(val) == 1):
                    var_datatype = const_types.DATATYPE_INTEGER
            if not var_datatype == const_types.DATATYPE_INTEGER:
                var_datatype = const_types.DATATYPE_BOOLEAN
        elif raw_type == "bool":
            var_datatype = const_types.DATATYPE_BOOLEAN
        elif raw_type == "float64" or raw_type == "float32":
            var_datatype = const_types.DATATYPE_FLOAT
        elif raw_type == "datetime64":
            var_datatype = const_types.DATATYPE_DATE
        elif raw_type == "object":
            var_datatype = const_types.DATATYPE_STRING

        return var_datatype

    def get_variable_type(self, feat_name):
        # Get the variable type based on data type and heuristics
        var_datatype = self.get_data_type(feat_name)
        var_vartype = const_types.VARTYPE_UNKNOWN

        # Variable type: categorical, continuous, binary
        if var_datatype == const_types.DATATYPE_BOOLEAN:
            var_vartype = const_types.VARTYPE_BINARY
        elif var_datatype == const_types.DATATYPE_STRING or var_datatype == const_types.DATATYPE_DATE:
            var_vartype = const_types.VARTYPE_CATEGORICAL
        elif var_datatype == const_types.DATATYPE_INTEGER or var_datatype == const_types.DATATYPE_FLOAT:
            # Distinguish int categorical variables (e.g. 1, 2, 3) from int continuous (e.g. 1, 2, ..., 1000)
            # Assuming categorical features have 10% or fewer unique values over the data set
            if self.get_percent_unique(feat_name) < 0.10:
                var_vartype = const_types.VARTYPE_CATEGORICAL
            else:
                var_vartype = const_types.VARTYPE_CONTINUOUS

        return var_vartype

    def get_percent_unique(self, feat_name):
        return float(len(self.data[feat_name].unique())) / self.data[feat_name].count()

    def get_count_unique(self, feat_name):
        return len(self.data[feat_name].unique())

    def get_freq_dictionary(self, feat1, feat2):
        freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
        freq_dictionary = {}
        colnames_unsorted = list(freq_table.columns)
        rownames_unsorted = list(freq_table.index)

        # Sort the row and column names for better printing
        colnames = sorted(colnames_unsorted)
        rownames = sorted(rownames_unsorted)

        for col in colnames:
            col_counts = {}
            for row in rownames:
                col_counts[str(row)] = int(freq_table[col][row])
            col_counts = OrderedDict(sorted(col_counts.items()))
            freq_dictionary[str(col)] = col_counts

        # Force the dictionary to stay sorted
        freq_dictionary = OrderedDict(sorted(freq_dictionary.items()))

        return freq_dictionary, str(colnames[0])

    def get_stats_by_category(self, categorical_feature, continuous_feature):
        unique_category_values = self.data[categorical_feature].unique()
        stats_by_categories = {}

        for category in unique_category_values:
            category_stats = {}

            # Get continuous values
            rows = self.data.loc[self.data[categorical_feature] == category][continuous_feature]
            category_stats["Maximum"] = float(rows.max())
            category_stats["Median"] = str("%.3f" % rows.median())
            category_stats["Mean"] = str("%.3f" % rows.mean())
            category_stats["Minimum"] = float(rows.min())
            category_stats["Skew"] = str("%.3f" % rows.skew())
            category_stats["Standard deviation"] = str("%.3f" % rows.std())

            # Add the stats for this category to the dictionary
            stats_by_categories[str(category)] = category_stats

        stats_by_categories_ordered = OrderedDict(sorted(stats_by_categories.items()))
        return stats_by_categories_ordered

    @staticmethod
    def get_stats_by_category_list():
        return ["Minimum", "Median", "Mean", "Maximum", "Standard deviation", "Skew"]

    def get_stats_by_category_flipped(self, continuous_feature, categorical_feature):
        unique_category_values = self.data[categorical_feature].unique()
        stats_list = self.get_stats_by_category_list()

        stats_by_categories = self.get_stats_by_category(categorical_feature, continuous_feature)
        stats_by_categories_flipped = {}

        for stat in stats_list:
            stat_values = {}
            for category_value in unique_category_values:
                # Get the statistic for this category value
                stat_values[str(category_value)] = stats_by_categories[str(category_value)][stat]
            stat_values_ordered = OrderedDict(sorted(stat_values.items()))
            stats_by_categories_flipped[stat] = stat_values_ordered

        return stats_by_categories_flipped

    def get_chisquared(self, feat1, feat2):
        freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
        if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
            return chi2_contingency(freq_table.dropna())

    def get_cramersv(self, feat1, feat2):
        freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
        if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
            chi2 = self.get_chisquared(feat1, feat2)[0]
            n = freq_table.sum().sum()
            return np.sqrt(chi2 / (n * (min(freq_table.shape) - 1)))

    def check_uniques_for_graphing(self, feat_name):
        return self.get_percent_unique(feat_name) < 0.2 or self.get_count_unique(feat_name) < 12

    def generate_interactions_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:
            interactions_collection = {}
            features = []

            feature_index = 0
            feature_names = list(self.data.columns.values)

            # Don't run any comparisons against the ID column
            if self.id_column in feature_names:
                feature_names.remove(self.id_column)

            statsforcategory = self.get_stats_by_category_list()

            # For each feature, get as much relevant info as possible
            for base_feat in feature_names:

                # Save the current feature to the collection
                feat_datatype = self.get_data_type(base_feat)
                feat_vartype = self.get_variable_type(base_feat)
                base_feature = Feature(feat_name=base_feat, feat_index=feature_index, feat_datatype=feat_datatype,
                                       feat_vartype=feat_vartype)
                features.append(base_feature)

                # Get a list of all other features
                other_features = feature_names.copy()
                other_features.remove(base_feat)

                # Create empty dictionaries to store comparisons of this field against all others
                scatterplots = {}
                correlations = {}
                covariances = {}
                boxplots = {}
                statsbycategory = {}
                statsbycategoryflipped = {}
                ztests = {}
                ttests = {}
                anova = {}
                stackedbarplots = {}
                chisquared = {}
                cramers = {}
                mantelhchi = {}
                frequencytable = {}
                frequencytable_firstrow = {}

                # Compare against all other features
                for compare_feat in other_features:

                    # Get the variable type and data type of both features
                    compare_vartype = self.get_variable_type(compare_feat)

                    # Case #1 - Base: continuous, compare: continuous
                    if feat_vartype == \
                            const_types.VARTYPE_CONTINUOUS and compare_vartype == const_types.VARTYPE_CONTINUOUS:

                        # Correlation
                        correlations[compare_feat] = str("%.3f" % float(self.data[[compare_feat, base_feat]]
                                                                        .corr()[compare_feat][base_feat]))

                        # Covariance
                        covariances[compare_feat] = str("%.3f" % float(self.data[[compare_feat, base_feat]]
                                                                       .cov()[compare_feat][base_feat]))

                        # Scatter plot
                        scatterplot = sns.regplot(x=base_feat, y=compare_feat,
                                                  data=self.data[[compare_feat, base_feat]])
                        scatterplots[compare_feat] = \
                            self.save_graph(scatterplot,
                                            filename=base_feat + "_" + compare_feat + paths.FILE_SCATTERPLOT)

                    # Case #2 - Base: categorical/ binary, compare: continuous
                    elif (feat_vartype == const_types.VARTYPE_CATEGORICAL
                          or feat_vartype == const_types.VARTYPE_BINARY) \
                            and compare_vartype == const_types.VARTYPE_CONTINUOUS:

                        # Don't plot if too many unique values
                        if self.check_uniques_for_graphing(base_feat):
                            statsbycategory[compare_feat] = self.get_stats_by_category(base_feat, compare_feat)

                            # box plot
                            boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="y",
                                                  data=self.data[[compare_feat, base_feat]])
                            boxplots[compare_feat] = \
                                self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)

                    # Case #3 - Base: continuous, compare: categorical/ binary
                    elif (
                                    compare_vartype == const_types.VARTYPE_CATEGORICAL or
                                    compare_vartype == const_types.VARTYPE_BINARY
                    ) and feat_vartype == const_types.VARTYPE_CONTINUOUS:

                        # Don't plot if too many unique values
                        if self.check_uniques_for_graphing(compare_feat):
                            statsbycategoryflipped[compare_feat] = \
                                self.get_stats_by_category_flipped(base_feat, compare_feat)

                            # Box plot
                            boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="h",
                                                  data=self.data[[compare_feat, base_feat]])
                            boxplots[compare_feat] = \
                                self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)

                    # Case #4 - Base: categorical/binary, compare: categorical/binary
                    elif (
                                    feat_vartype == const_types.VARTYPE_CATEGORICAL or
                                    feat_vartype == const_types.VARTYPE_BINARY
                    ) and (
                                    compare_vartype == const_types.VARTYPE_CATEGORICAL or
                                    compare_vartype == const_types.VARTYPE_BINARY):

                        if self.check_uniques_for_graphing(base_feat) and self.check_uniques_for_graphing(compare_feat):

                            # Bar chart (x = base, y = # occ, color = compare)
                            barchart = sns.countplot(x=base_feat, hue=compare_feat,
                                                     data=self.data[[base_feat, compare_feat]].dropna())
                            stackedbarplots[compare_feat] = \
                                self.save_graph(barchart, filename=base_feat + "_" + compare_feat + paths.FILE_BARCHART)

                            # Chi-Squared
                            chi_results = self.get_chisquared(base_feat, compare_feat)
                            if chi_results is not None:
                                (chi2, p, dof, ex) = chi_results
                                if p <= 0.001:
                                    p_sig = "p ≤ 0.001***"
                                elif p <= 0.01:
                                    p_sig = "p ≤ 0.01**"
                                elif p <= 0.05:
                                    p_sig = "p ≤ 0.05*"
                                else:
                                    p_sig = "ns"
                                chisquared[compare_feat] = str("%.3f, %s (p=%.7f)" % (chi2, p_sig, p))

                            # Cramer's V
                            cramersvstat = self.get_cramersv(base_feat, compare_feat)
                            if cramersvstat is not None:
                                cramers[compare_feat] = str("%.3f" % cramersvstat)

                        # Display frequency table, limit number of results
                        if self.get_count_unique(base_feat) <= 10 and self.get_count_unique(compare_feat) <= 50:
                            # Frequency table
                            frequency_dictionary, first_row_key = self.get_freq_dictionary(base_feat, compare_feat)
                            frequencytable[compare_feat] = frequency_dictionary
                            frequencytable_firstrow[compare_feat] = first_row_key

                # Create interaction object comparing this feature to all others
                interaction = Interaction(feat_name=base_feat,
                                          feat_index=feature_index,
                                          other_features=other_features,
                                          scatterplots=scatterplots,
                                          correlations=correlations,
                                          covariances=covariances,
                                          boxplots=boxplots,
                                          statsbycategory=statsbycategory,
                                          statsbycategoryflipped=statsbycategoryflipped,
                                          statsforcategory=statsforcategory,
                                          ztests=ztests,
                                          ttests=ttests,
                                          anova=anova,
                                          stackedbarplots=stackedbarplots,
                                          chisquared=chisquared,
                                          cramers=cramers,
                                          mantelhchi=mantelhchi,
                                          frequency_table=frequencytable,
                                          frequencytable_firstrow=frequencytable_firstrow)

                # Add to the collection of interactions
                interactions_collection[base_feat] = interaction
                feature_index += 1

            # Create interactions object to represent the entire collection
            interactions = Interactions(name=self.title,
                                        features=features,
                                        feature_interactions=interactions_collection)
            interactions_json = jsonpickle.encode(interactions)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=interactions_json, suffix=paths.INTERACTIONS_SUFFIX)

    def get_error_msg(self):
        return self.error_code

    def save_json(self, json_to_write, suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        file = open(os.path.join(folder_path, self.title, suffix), 'w')
        file.write(json_to_write)
        file.close()

    def load_summary_json(self):
        return self.load_json(paths.SUMMARY_SUFFIX)

    def load_features_json(self):
        return self.load_json(paths.FEATURES_SUFFIX)

    def load_interactions_json(self):
        return self.load_json(paths.INTERACTIONS_SUFFIX)

    def load_json(self, json_suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        absolute_filename = os.path.join(folder_path, self.title, json_suffix)

        # Check if the JSON file exists and if not, generate it
        if not os.path.isfile(absolute_filename):
            if json_suffix == paths.SUMMARY_SUFFIX:
                self.generate_summary_json()
            elif json_suffix == paths.FEATURES_SUFFIX:
                self.generate_features_json()
            elif json_suffix == paths.INTERACTIONS_SUFFIX:
                self.generate_interactions_json()

        # Read serialized JSON file
        if os.path.isfile(absolute_filename):
            with open(absolute_filename, 'r') as serialized_file:
                json_str = serialized_file.read()
                deserialized_json = jsonpickle.decode(json_str)
            return deserialized_json
