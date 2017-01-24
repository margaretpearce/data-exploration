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

    # def get_freq_dictionary(self, feat1, feat2):
    #     freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
    #     freq_dictionary = {}
    #     colnames_unsorted = list(freq_table.columns)
    #     rownames_unsorted = list(freq_table.index)
    #
    #     # Sort the row and column names for better printing
    #     colnames = sorted(colnames_unsorted)
    #     rownames = sorted(rownames_unsorted)
    #
    #     for col in colnames:
    #         col_counts = {}
    #         for row in rownames:
    #             col_counts[str(row)] = int(freq_table[col][row])
    #         col_counts = OrderedDict(sorted(col_counts.items()))
    #         freq_dictionary[str(col)] = col_counts
    #
    #     # Force the dictionary to stay sorted
    #     freq_dictionary = OrderedDict(sorted(freq_dictionary.items()))
    #
    #     return freq_dictionary, str(colnames[0])
    #
    # def get_stats_by_category(self, categorical_feature, continuous_feature):
    #     unique_category_values = self.data[categorical_feature].unique()
    #     stats_by_categories = {}
    #
    #     for category in unique_category_values:
    #         category_stats = {}
    #
    #         # Get continuous values
    #         rows = self.data.loc[self.data[categorical_feature] == category][continuous_feature]
    #         category_stats["Maximum"] = float(rows.max())
    #         category_stats["Median"] = str("%.3f" % rows.median())
    #         category_stats["Mean"] = str("%.3f" % rows.mean())
    #         category_stats["Minimum"] = float(rows.min())
    #         category_stats["Skew"] = str("%.3f" % rows.skew())
    #         category_stats["Standard deviation"] = str("%.3f" % rows.std())
    #
    #         # Add the stats for this category to the dictionary
    #         stats_by_categories[str(category)] = category_stats
    #
    #     stats_by_categories_ordered = OrderedDict(sorted(stats_by_categories.items()))
    #     return stats_by_categories_ordered
    #
    # @staticmethod
    # def get_stats_by_category_list():
    #     return ["Minimum", "Median", "Mean", "Maximum", "Standard deviation", "Skew"]
    #
    # def get_stats_by_category_flipped(self, continuous_feature, categorical_feature):
    #     unique_category_values = self.data[categorical_feature].unique()
    #     stats_list = self.get_stats_by_category_list()
    #
    #     stats_by_categories = self.get_stats_by_category(categorical_feature, continuous_feature)
    #     stats_by_categories_flipped = {}
    #
    #     for stat in stats_list:
    #         stat_values = {}
    #         for category_value in unique_category_values:
    #             # Get the statistic for this category value
    #             stat_values[str(category_value)] = stats_by_categories[str(category_value)][stat]
    #         stat_values_ordered = OrderedDict(sorted(stat_values.items()))
    #         stats_by_categories_flipped[stat] = stat_values_ordered
    #
    #     return stats_by_categories_flipped
    #
    # def get_chisquared(self, feat1, feat2):
    #     freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
    #     if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
    #         return chi2_contingency(freq_table.dropna())
    #
    # def get_cramersv(self, feat1, feat2):
    #     freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
    #     if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
    #         chi2 = self.get_chisquared(feat1, feat2)[0]
    #         n = freq_table.sum().sum()
    #         return np.sqrt(chi2 / (n * (min(freq_table.shape) - 1)))

    def check_uniques_for_graphing(self, feat_name):
        return self.get_percent_unique(feat_name) < 0.2 or self.get_count_unique(feat_name) < 12

    # def generate_interactions_json(self):
    #     load_success = True
    #
    #     # Check if the data file exists, and if so, load the data as needed
    #     if self.data is None and os.path.isfile(self.filepath):
    #         load_success = self.load_data()
    #
    #     if load_success:
    #         interactions_collection = {}
    #         features = []
    #
    #         feature_index = 0
    #         feature_names = list(self.data.columns.values)
    #
    #         # Don't run any comparisons against the ID column
    #         if self.id_column in feature_names:
    #             feature_names.remove(self.id_column)
    #
    #         statsforcategory = self.get_stats_by_category_list()
    #
    #         # For each feature, get as much relevant info as possible
    #         for base_feat in feature_names:
    #
    #             # Save the current feature to the collection
    #             feat_datatype = self.get_data_type(base_feat)
    #             feat_vartype = self.get_variable_type(base_feat)
    #             base_feature = Feature(feat_name=base_feat, feat_index=feature_index, feat_datatype=feat_datatype,
    #                                    feat_vartype=feat_vartype)
    #             features.append(base_feature)
    #
    #             # Get a list of all other features
    #             other_features = feature_names.copy()
    #             other_features.remove(base_feat)
    #
    #             # Create empty dictionaries to store comparisons of this field against all others
    #             scatterplots = {}
    #             correlations = {}
    #             covariances = {}
    #             boxplots = {}
    #             statsbycategory = {}
    #             statsbycategoryflipped = {}
    #             ztests = {}
    #             ttests = {}
    #             anova = {}
    #             stackedbarplots = {}
    #             chisquared = {}
    #             cramers = {}
    #             mantelhchi = {}
    #             frequencytable = {}
    #             frequencytable_firstrow = {}
    #
    #             # Compare against all other features
    #             for compare_feat in other_features:
    #
    #                 # Get the variable type and data type of both features
    #                 compare_vartype = self.get_variable_type(compare_feat)
    #
    #                 # Case #1 - Base: continuous, compare: continuous
    #                 if feat_vartype == \
    #                         const_types.VARTYPE_CONTINUOUS and compare_vartype == const_types.VARTYPE_CONTINUOUS:
    #
    #                     # Correlation
    #                     correlations[compare_feat] = str("%.3f" % float(self.data[[compare_feat, base_feat]]
    #                                                                     .corr()[compare_feat][base_feat]))
    #
    #                     # Covariance
    #                     covariances[compare_feat] = str("%.3f" % float(self.data[[compare_feat, base_feat]]
    #                                                                    .cov()[compare_feat][base_feat]))
    #
    #                     # Scatter plot
    #                     scatterplot = sns.regplot(x=base_feat, y=compare_feat,
    #                                               data=self.data[[compare_feat, base_feat]])
    #                     scatterplots[compare_feat] = \
    #                         self.save_graph(scatterplot,
    #                                         filename=base_feat + "_" + compare_feat + paths.FILE_SCATTERPLOT)
    #
    #                 # Case #2 - Base: categorical/ binary, compare: continuous
    #                 elif (feat_vartype == const_types.VARTYPE_CATEGORICAL
    #                       or feat_vartype == const_types.VARTYPE_BINARY) \
    #                         and compare_vartype == const_types.VARTYPE_CONTINUOUS:
    #
    #                     # Don't plot if too many unique values
    #                     if self.check_uniques_for_graphing(base_feat):
    #                         statsbycategory[compare_feat] = self.get_stats_by_category(base_feat, compare_feat)
    #
    #                         # box plot
    #                         boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="y",
    #                                               data=self.data[[compare_feat, base_feat]])
    #
    #                         if self.get_count_unique(base_feat) > 8:
    #                             boxplot.set_xticklabels(labels=boxplot.get_xticklabels(), rotation=45)
    #
    #                         boxplots[compare_feat] = \
    #                             self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)
    #
    #                 # Case #3 - Base: continuous, compare: categorical/ binary
    #                 elif (
    #                                 compare_vartype == const_types.VARTYPE_CATEGORICAL or
    #                                 compare_vartype == const_types.VARTYPE_BINARY
    #                 ) and feat_vartype == const_types.VARTYPE_CONTINUOUS:
    #
    #                     # Don't plot if too many unique values
    #                     if self.check_uniques_for_graphing(compare_feat):
    #                         statsbycategoryflipped[compare_feat] = \
    #                             self.get_stats_by_category_flipped(base_feat, compare_feat)
    #
    #                         # Box plot
    #                         boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="h",
    #                                               data=self.data[[compare_feat, base_feat]])
    #
    #                         boxplots[compare_feat] = \
    #                             self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)
    #
    #                 # Case #4 - Base: categorical/binary, compare: categorical/binary
    #                 elif (
    #                                 feat_vartype == const_types.VARTYPE_CATEGORICAL or
    #                                 feat_vartype == const_types.VARTYPE_BINARY
    #                 ) and (
    #                                 compare_vartype == const_types.VARTYPE_CATEGORICAL or
    #                                 compare_vartype == const_types.VARTYPE_BINARY):
    #
    #                     if self.check_uniques_for_graphing(base_feat) and self.check_uniques_for_graphing(compare_feat):
    #
    #                         # Bar chart (x = base, y = # occ, color = compare)
    #                         barchart = sns.countplot(x=base_feat, hue=compare_feat,
    #                                                  data=self.data[[base_feat, compare_feat]].dropna())
    #
    #                         if self.get_count_unique(base_feat) > 8:
    #                             barchart.set_xticklabels(labels=barchart.get_xticklabels(), rotation=45)
    #
    #                         stackedbarplots[compare_feat] = \
    #                             self.save_graph(barchart, filename=base_feat + "_" + compare_feat + paths.FILE_BARCHART)
    #
    #                         # Chi-Squared
    #                         chi_results = self.get_chisquared(base_feat, compare_feat)
    #                         if chi_results is not None:
    #                             (chi2, p, dof, ex) = chi_results
    #                             if p <= 0.001:
    #                                 p_sig = "p ≤ 0.001***"
    #                             elif p <= 0.01:
    #                                 p_sig = "p ≤ 0.01**"
    #                             elif p <= 0.05:
    #                                 p_sig = "p ≤ 0.05*"
    #                             else:
    #                                 p_sig = "ns"
    #                             chisquared[compare_feat] = str("%.3f, %s (p=%.7f)" % (chi2, p_sig, p))
    #
    #                         # Cramer's V
    #                         cramersvstat = self.get_cramersv(base_feat, compare_feat)
    #                         if cramersvstat is not None:
    #                             cramers[compare_feat] = str("%.3f" % cramersvstat)
    #
    #                     # Display frequency table, limit number of results
    #                     if self.get_count_unique(base_feat) <= 30 and self.get_count_unique(compare_feat) <= 50:
    #                         # Frequency table
    #                         frequency_dictionary, first_row_key = self.get_freq_dictionary(base_feat, compare_feat)
    #                         frequencytable[compare_feat] = frequency_dictionary
    #                         frequencytable_firstrow[compare_feat] = first_row_key
    #
    #             # Create interaction object comparing this feature to all others
    #             interaction = Interaction(feat_name=base_feat,
    #                                       feat_index=feature_index,
    #                                       other_features=other_features,
    #                                       scatterplots=scatterplots,
    #                                       correlations=correlations,
    #                                       covariances=covariances,
    #                                       boxplots=boxplots,
    #                                       statsbycategory=statsbycategory,
    #                                       statsbycategoryflipped=statsbycategoryflipped,
    #                                       statsforcategory=statsforcategory,
    #                                       ztests=ztests,
    #                                       ttests=ttests,
    #                                       anova=anova,
    #                                       stackedbarplots=stackedbarplots,
    #                                       chisquared=chisquared,
    #                                       cramers=cramers,
    #                                       mantelhchi=mantelhchi,
    #                                       frequency_table=frequencytable,
    #                                       frequencytable_firstrow=frequencytable_firstrow)
    #
    #             # Add to the collection of interactions
    #             interactions_collection[base_feat] = interaction
    #             feature_index += 1
    #
    #         # Create interactions object to represent the entire collection
    #         interactions = Interactions(name=self.title,
    #                                     features=features,
    #                                     feature_interactions=interactions_collection)
    #         interactions_json = jsonpickle.encode(interactions)
    #
    #         # Save the serialized JSON to a file
    #         self.save_json(json_to_write=interactions_json, suffix=paths.INTERACTIONS_SUFFIX)

    def get_error_msg(self):
        return self.error_code

    def save_json(self, json_to_write, suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        file = open(os.path.join(folder_path, self.title, suffix), 'w')
        file.write(json_to_write)
        file.close()

    def load_json(self, json_suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        absolute_filename = os.path.join(folder_path, self.title, json_suffix)

        # Check if the JSON file exists and if not, generate it
        if not os.path.isfile(absolute_filename):
            return None

        # Read serialized JSON file
        if os.path.isfile(absolute_filename):
            with open(absolute_filename, 'r') as serialized_file:
                json_str = serialized_file.read()
                deserialized_json = jsonpickle.decode(json_str)
            return deserialized_json
