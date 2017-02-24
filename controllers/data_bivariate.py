import os
from collections import OrderedDict
import jsonpickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from configuration import const_types
from configuration import paths
from controllers.data_driver import DataDriver
from model.feature import Feature
from model.interaction import Interaction
from model.interactions import Interactions


class DataBivariate(DataDriver):
    def __init__(self, selected_dataset):
        DataDriver.__init__(self, selected_dataset)

    def load_interactions_json(self):
        interactions_json = self.load_json(paths.INTERACTIONS_SUFFIX)

        # If the file doesn't exist, generate it
        if interactions_json is None:
            self.generate_interactions_json()
            interactions_json = self.load_json(paths.INTERACTIONS_SUFFIX)

        # Return the JSON
        return interactions_json

    def generate_interactions_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:
            interactions_collection = {}
            features = []

            feature_index = 0
            feature_names = self.get_features_list()

            # Don't run any comparisons against the ID column
            if self.id_column in feature_names:
                feature_names.remove(self.id_column)

            statsforcategory = self.get_stats_by_category_list()

            # For each feature, get as much relevant info as possible
            for base_feat in feature_names:

                # Save the current feature to the collection
                base_feature = self.get_base_feature(base_feat, feature_index)
                features.append(base_feature)

                interaction = self.get_feature_interactions(base_feat, feature_index, feature_names, statsforcategory)

                # Add to the collection of interactions
                if not self.check_feature_for_removal(interaction):
                    interactions_collection[base_feat] = interaction
                    feature_index += 1
                else:
                    features.remove(base_feature)

            # Create interactions object to represent the entire collection
            interactions = Interactions(name=self.title,
                                        features=features,
                                        feature_interactions=interactions_collection)
            interactions_json = jsonpickle.encode(interactions)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=interactions_json, suffix=paths.INTERACTIONS_SUFFIX)

    def get_feature_interactions(self, base_feat, feature_index, feature_names, statsforcategory):
        other_features = feature_names.copy()
        other_features.remove(base_feat)

        # Create empty dictionaries to store comparisons of this field against all others
        scatterplots = {}
        correlations = {}
        covariances = {}
        boxplots = {}
        statsbycategory = {}
        statsbycategoryflipped = {}
        stackedbarplots = {}
        chisquared = {}
        cramers = {}
        mantelhchi = {}
        frequencytable = {}
        frequencytable_firstrow = {}

        # Compare against all other features
        for compare_feat in other_features:
                correlations[compare_feat] = self.get_correlation(base_feat, compare_feat)
                covariances[compare_feat] = self.get_covariance(base_feat, compare_feat)
                scatterplots[compare_feat] = self.get_scatterplot(base_feat, compare_feat)
                statsbycategory[compare_feat] = self.get_stats_by_category(base_feat, compare_feat)
                boxplots[compare_feat] = self.get_boxplot(base_feat, compare_feat)
                statsbycategoryflipped[compare_feat] = self.get_stats_by_category_flipped(base_feat, compare_feat)
                stackedbarplots[compare_feat] = self.get_colored_countplot(base_feat, compare_feat)
                chisquared[compare_feat] = self.get_chisquared_formatted(base_feat, compare_feat)
                cramers[compare_feat] = self.format_rounded_string(self.get_cramersv(base_feat, compare_feat))
                frequency_dictionary, first_row_key = self.get_freq_dictionary(base_feat, compare_feat)
                frequencytable[compare_feat] = frequency_dictionary
                frequencytable_firstrow[compare_feat] = first_row_key

                # Remove nulls
                if correlations[compare_feat] is None:
                    correlations.pop(compare_feat)
                if covariances[compare_feat] is None:
                    covariances.pop(compare_feat)
                if scatterplots[compare_feat] is None:
                    scatterplots.pop(compare_feat)
                if statsbycategory[compare_feat] is None:
                    statsbycategory.pop(compare_feat)
                if boxplots[compare_feat] is None:
                    boxplots.pop(compare_feat)
                if statsbycategoryflipped[compare_feat] is None:
                    statsbycategoryflipped.pop(compare_feat)
                if stackedbarplots[compare_feat] is None:
                    stackedbarplots.pop(compare_feat)
                if chisquared[compare_feat] is None:
                    chisquared.pop(compare_feat)
                if cramers[compare_feat] is None:
                    cramers.pop(compare_feat)
                if frequencytable[compare_feat] is None:
                    frequencytable.pop(compare_feat)
                if frequencytable_firstrow[compare_feat] is None:
                    frequencytable_firstrow.pop(compare_feat)

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
                                  ztests=None,
                                  ttests=None,
                                  anova=None,
                                  stackedbarplots=stackedbarplots,
                                  chisquared=chisquared,
                                  cramers=cramers,
                                  mantelhchi=mantelhchi,
                                  frequency_table=frequencytable,
                                  frequencytable_firstrow=frequencytable_firstrow)
        return interaction

    def get_base_feature(self, base_feat, feature_index):
        feat_datatype = self.get_data_type(base_feat)
        feat_vartype = self.get_variable_type(base_feat)
        base_feature = Feature(feat_name=base_feat, feat_index=feature_index, feat_datatype=feat_datatype,
                               feat_vartype=feat_vartype)
        return base_feature

    @staticmethod
    def check_feature_for_removal(interaction):
        # Return true if empty, else false
        return not interaction.scatterplots and \
            not interaction.correlations and \
            not interaction.covariances and \
            not interaction.boxplots and \
            not interaction.statsbycategory and \
            not interaction.statsbycategoryflipped and \
            not interaction.stackedbarplots and \
            not interaction.chisquared and \
            not interaction.cramers and \
            not interaction.mantelhchi and \
            not interaction.frequency_table

    def get_correlation(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if base_vartype == const_types.VARTYPE_CONTINUOUS and compare_vartype == const_types.VARTYPE_CONTINUOUS:
            return float(self.data[[compare_feat, base_feat]].corr()[compare_feat][base_feat])

        return None

    def get_covariance(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if base_vartype == const_types.VARTYPE_CONTINUOUS and compare_vartype == const_types.VARTYPE_CONTINUOUS:
            return float(self.data[[compare_feat, base_feat]].cov()[compare_feat][base_feat])

        return None

    def get_scatterplot(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if base_vartype == const_types.VARTYPE_CONTINUOUS and compare_vartype == const_types.VARTYPE_CONTINUOUS:
            scatterplot = sns.regplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]])
            return self.save_graph(scatterplot, filename=base_feat + "_" + compare_feat + paths.FILE_SCATTERPLOT)

        return None

    def get_boxplot(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
           compare_vartype == const_types.VARTYPE_CONTINUOUS:
                return self.get_vertical_boxplot(base_feat, compare_feat)

        if (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY) and \
           base_vartype == const_types.VARTYPE_CONTINUOUS:
                return self.get_horizontal_boxplot(base_feat, compare_feat)

        return None

    def get_vertical_boxplot(self, base_feat, compare_feat):
        if self.check_uniques_for_graphing(base_feat):
            boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="y", data=self.data[[compare_feat, base_feat]])

            if self.get_count_unique(base_feat) > 8:
                boxplot.set_xticklabels(labels=boxplot.get_xticklabels(), rotation=45)

            return self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)

        return None

    def get_horizontal_boxplot(self, base_feat, compare_feat):
        if self.check_uniques_for_graphing(compare_feat):
            boxplot = sns.boxplot(x=base_feat, y=compare_feat, orient="h", data=self.data[[compare_feat, base_feat]])

            if self.get_count_unique(base_feat) > 8:
                boxplot.set_xticklabels(labels=boxplot.get_xticklabels(), rotation=45)

            return self.save_graph(boxplot, filename=base_feat + "_" + compare_feat + paths.FILE_BOXCHART)

        return None

    def get_colored_countplot(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
           (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY):

            if self.check_uniques_for_graphing(base_feat) and self.check_uniques_for_graphing(compare_feat):

                # Bar chart (x = base, y = # occ, color = compare)
                barchart = sns.countplot(x=base_feat, hue=compare_feat,
                                         data=self.data[[base_feat, compare_feat]].dropna())

                if self.get_count_unique(base_feat) > 8:
                    barchart.set_xticklabels(labels=barchart.get_xticklabels(), rotation=45)

                return self.save_graph(barchart, filename=base_feat + "_" + compare_feat + paths.FILE_BARCHART)

        return None

    def get_chisquared_formatted(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
                (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY):

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
                return str("%.3f, %s (p=%.7f)" % (chi2, p_sig, p))

        return None

    def get_freq_dictionary(self, feat1, feat2):
        base_vartype = self.get_variable_type(feat1)
        compare_vartype = self.get_variable_type(feat2)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
                (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY):

            if self.get_count_unique(feat1) <= 30 and self.get_count_unique(feat2) <= 50:
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

        return None, None

    def get_stats_by_category(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
           compare_vartype == const_types.VARTYPE_CONTINUOUS:

            if self.check_uniques_for_graphing(base_feat):
                unique_category_values = self.data[base_feat].unique()
                stats_by_categories = {}

                for category in unique_category_values:
                    category_stats = {}

                    # Get continuous values
                    rows = self.data.loc[self.data[base_feat] == category][compare_feat]
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

        return None

    @staticmethod
    def get_stats_by_category_list():
        return ["Minimum", "Median", "Mean", "Maximum", "Standard deviation", "Skew"]

    def get_stats_by_category_flipped(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY) and \
           base_vartype == const_types.VARTYPE_CONTINUOUS:

            if self.check_uniques_for_graphing(compare_feat):
                unique_category_values = self.data[compare_feat].unique()
                stats_list = self.get_stats_by_category_list()

                stats_by_categories = self.get_stats_by_category(compare_feat, base_feat)
                stats_by_categories_flipped = {}

                for stat in stats_list:
                    stat_values = {}
                    for category_value in unique_category_values:
                        # Get the statistic for this category value
                        stat_values[str(category_value)] = stats_by_categories[str(category_value)][stat]
                    stat_values_ordered = OrderedDict(sorted(stat_values.items()))
                    stats_by_categories_flipped[stat] = stat_values_ordered

                return stats_by_categories_flipped

        return None

    def get_chisquared(self, feat1, feat2):
        base_vartype = self.get_variable_type(feat1)
        compare_vartype = self.get_variable_type(feat2)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
                (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY):

            freq_table = pd.crosstab(self.data[feat1], self.data[feat2])
            if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
                return chi2_contingency(freq_table.dropna())

        return None

    def get_cramersv(self, base_feat, compare_feat):
        base_vartype = self.get_variable_type(base_feat)
        compare_vartype = self.get_variable_type(compare_feat)

        if (base_vartype == const_types.VARTYPE_CATEGORICAL or base_vartype == const_types.VARTYPE_BINARY) and \
                (compare_vartype == const_types.VARTYPE_CATEGORICAL or compare_vartype == const_types.VARTYPE_BINARY):

            freq_table = pd.crosstab(self.data[base_feat], self.data[compare_feat])
            if len(list(filter(lambda x: x < 5, freq_table.values.flatten()))) == 0:
                chi2 = self.get_chisquared(base_feat, compare_feat)[0]
                n = freq_table.sum().sum()
                return np.sqrt(chi2 / (n * (min(freq_table.shape) - 1)))

        return None
