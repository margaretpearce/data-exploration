import pandas as pd
import os
import jsonpickle
import paths
import seaborn as sns
from scipy.stats import *
from Summary import DataSummary
from Interactions import Interactions
from Interaction import Interaction
from Feature import Feature
from Features import Features

SUMMARY_SUFFIX = "_summary.json"
FEATURES_SUFFIX = "_features.json"
INTERACTIONS_SUFFIX = "_interactions.json"


class DataDriver:
    def __init__(self, datafile, title, idcolumn=None, labelcolumn=None):
        self.file = datafile
        self.filepath = os.path.join(paths.EXAMPLES_FOLDER, self.file)
        self.title = title
        self.id_column = idcolumn
        self.label_column = labelcolumn

        # Other class variables
        self.data = None

        # Check if the data file exists, and if so, load the data
        if os.path.isfile(self.filepath):
            self.load_data()

    def load_data(self):
        # Load the data into a Pandas DataFrame
        if str(self.file).endswith("csv"):
            self.data = pd.read_csv(self.filepath)
        elif str(self.file).endswith("xls") or str(self.file).endswith("xlsx"):
            self.data = pd.read_excel(self.filepath)
        elif str(self.file).endswith("json"):
            self.data = pd.read_json(self.filepath)

    def generate_summary_json(self):
        # Get summary stats about the data and serialize it as JSON
        num_records = self.data.shape[0]
        num_features = self.data.shape[1]
        index_column = self.id_column
        label_column = self.label_column

        # Count the number of columns missing for each row
        count_missing = self.data.apply(lambda x: sum(x.isnull().values), axis=1)
        self.data["num_missing"] = pd.Series(count_missing)

        num_rows_no_missing = int(sum(self.data["num_missing"] == 0))
        num_rows_one_missing = int(sum(self.data["num_missing"] == 1))
        num_rows_two_missing = int(sum(self.data["num_missing"] == 2))
        num_rows_three_more_missing = int(sum(self.data["num_missing"] >= 3))

        # Sample data (five rows)
        features_list = list(self.data.columns.values)
        sample_list = self.data.head()[features_list].values.tolist()

        summary = DataSummary(name=self.title,
                              num_records=num_records,
                              num_features=num_features,
                              index_column=index_column,
                              label_column=label_column,
                              rows_no_missing=num_rows_no_missing,
                              rows_one_missing=num_rows_one_missing,
                              rows_two_missing=num_rows_two_missing,
                              rows_three_more_missing=num_rows_three_more_missing,
                              features_list=features_list,
                              sample_list=sample_list
                              )
        summary_json = jsonpickle.encode(summary)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + SUMMARY_SUFFIX)), 'w')
        file.write(summary_json)
        file.close()

    def generate_features_json(self):
        features_collection = []
        feature_index = 0

        # For each feature, get as much relevant info as possible
        for var_name in self.data.columns.values:
            # Common for all field types
            var_type = None
            raw_type = str(self.data[var_name].dtype)

            if raw_type == "int64":
                # Check if it's really a boolean
                unique_vals = self.data[var_name].unique()
                for val in unique_vals:
                    if not (int(val) == 0 or int(val) == 1):
                        var_type = "Integer"
                if not var_type == "Integer":
                    var_type = "Boolean"
            elif raw_type == "float64":
                var_type = "Float"
            elif raw_type == "datetime64":
                var_type = "Date"
            elif raw_type == "object":
                var_type = "String"

            var_count = int(self.data[var_name].count())

            missing_count = int(self.data[var_name].isnull().sum())
            missing_percent = missing_count / float(var_count)
            var_missing = str("%s (%.3f%%)" % (missing_count, missing_percent))

            var_unique = int(len(self.data[var_name].unique()))

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

                mode = self.data[var_name].mode()
                if mode is not None:
                    var_mode = ""
                    for m in mode:
                        var_mode = var_mode + str(m) + " "

            # Compute non-numeric stats
            else:
                var_mostcommon = str("%s (%d)" %
                                     (self.data[var_name].value_counts().idxmax(),
                                      self.data[var_name].value_counts().max()))
                var_leastcommon = str("%s (%d)" %
                                     (self.data[var_name].value_counts().idxmin(),
                                      self.data[var_name].value_counts().min()))

            # Histogram/ countplot
            if self.data[var_name].dtype in ['int64', 'float64']:
                hist_plot = sns.distplot(self.data[var_name].dropna(), bins=None, hist=True, kde=False, rug=False)
                full_url = os.path.join(paths.EXAMPLES_FOLDER, str(var_name + "_hist.png"))
                fig = hist_plot.get_figure()
                fig.savefig(full_url)   # Save the histogram
                sns.plt.clf()   # Clear the figure to prepare for the next plot
                graph_histogram = paths.EXAMPLES_RELATIVE + str(var_name + "_hist.png")    # Relative histogram URL
            else:
                countplot = sns.countplot(y=self.data[var_name].dropna())

                if (var_unique / float(var_count)) > 0.5:
                    countplot.set(ylabel='')
                    countplot.set(yticklabels=[])
                    countplot.yaxis.set_visible(False)

                full_url = os.path.join(paths.EXAMPLES_FOLDER, str(var_name + "_countplot.png"))
                fig = countplot.get_figure()
                fig.savefig(full_url)
                sns.plt.clf()   # Clear the figure to prepare for the next plot

                # Return the relative URL to the histogram
                graph_countplot = paths.EXAMPLES_RELATIVE + str(var_name + "_countplot.png")

            feature = Feature(feat_name=var_name,
                              feat_index=feature_index,
                              feat_type=var_type,
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
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + FEATURES_SUFFIX)), 'w')
        file.write(features_json)
        file.close()

    def generate_interactions_json(self):
        interactions_collection = {}
        features = []

        feature_index = 0
        feature_names = list(self.data.columns.values)

        # For each feature, get as much relevant info as possible
        for base_feat in feature_names:
            # Save the current feature to the collection
            base_feature = Feature(feat_name=base_feat, feat_index=feature_index)
            features.append(base_feature)

            # Get a list of all other features
            other_features = feature_names.copy()
            other_features.remove(base_feat)

            # Create empty dictionaries to store comparisons of this field against all others
            scatterplots={}
            correlations={}
            covariances={}
            boxplots={}
            ztests={}
            ttests={}
            anova={}
            stackedbarplots={}
            chisquared={}
            craters={}
            mantelhchi={}

            # Compare against all other features
            for compare_feat in other_features:
                base_is_numeric = self.data[base_feat].dtype in ['int64', 'float64']
                compare_is_numeric = self.data[compare_feat].dtype in ['int64', 'float64']

                base_is_categorical = 1.*self.data[base_feat].nunique()/self.data[base_feat].count() < 0.05
                compare_is_categorical = 1.*self.data[compare_feat].nunique()/self.data[compare_feat].count() < 0.05

                base_all_unique = 1.*self.data[base_feat].nunique()/self.data[base_feat].count() > 0.95
                compare_all_unique = 1.*self.data[compare_feat].nunique()/self.data[compare_feat].count() > 0.95

                # Numeric + numeric: Get numeric stats
                if base_is_numeric and compare_is_numeric:
                    # Correlation
                    correlations[compare_feat] = float(self.data[[compare_feat, base_feat]]
                                                       .corr()[compare_feat][base_feat])

                    # Covariance
                    covariances[compare_feat] = float(self.data[[compare_feat, base_feat]]
                                                      .cov()[compare_feat][base_feat])
                # Continous + continuous
                if base_is_numeric and compare_is_numeric:
                    # Scatter plot
                    scatterplot = sns.regplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]])
                    full_url = os.path.join(paths.EXAMPLES_FOLDER, str("graphs/" + base_feat + "_" + compare_feat + "_scatter.png"))
                    fig = scatterplot.get_figure()
                    fig.savefig(full_url)
                    sns.plt.clf()   # Clear the figure to prepare for the next plot
                    scatterplots[compare_feat] = paths.EXAMPLES_RELATIVE + \
                                                 str("graphs/" + base_feat + "_" + compare_feat + "_scatter.png")
                elif base_is_categorical and not compare_is_categorical:
                    # Only do the plot if there aren't too many unique values for base
                    if compare_is_numeric:
                        # Swarm plot
                        boxplot = sns.swarmplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]]);
                        full_url = os.path.join(paths.EXAMPLES_FOLDER, str("graphs/" + base_feat + "_" + compare_feat + "_box.png"))
                        fig = boxplot.get_figure()
                        fig.savefig(full_url)
                        sns.plt.clf()   # Clear the figure to prepare for the next plot
                        boxplots[compare_feat] = paths.EXAMPLES_RELATIVE + \
                                                 str("graphs/" + base_feat + "_" + compare_feat + "_box.png")

                # elif base_is_numeric and not compare_is_numeric:
                #     # Numeric + non-numeric
                #     # Box plots
                #     boxplot = sns.swarmplot(x=compare_feat, y=base_feat, data=self.data[[compare_feat, base_feat]]);
                #     # boxplot = sns.boxplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]])
                #     full_url = os.path.join(paths.EXAMPLES_FOLDER, str(base_feat + "_" + compare_feat + "_box.png"))
                #     fig = boxplot.get_figure()
                #     fig.savefig(full_url)
                #     sns.plt.clf()   # Clear the figure to prepare for the next plot
                #     boxplots[compare_feat] = paths.EXAMPLES_RELATIVE + \
                #                                  str(base_feat + "_" + compare_feat + "_box.png")
                #
                #     # z-test
                #     # t-test
                #     # ttest = ttest_ind(self.data[compare_feat].tolist(), self.data[base_feat].tolist())
                #     # ttests[compare_feat] = str(ttest)
                #
                #     # ANOVA
                #     break
                # elif not base_is_numeric and compare_is_numeric:
                #     # Numeric + non-numeric
                #     # Box plots
                #     boxplot = sns.swarmplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]]);
                #     # boxplot = sns.boxplot(x=base_feat, y=compare_feat, data=self.data[[compare_feat, base_feat]])
                #     full_url = os.path.join(paths.EXAMPLES_FOLDER, str(base_feat + "_" + compare_feat + "_box.png"))
                #     fig = boxplot.get_figure()
                #     fig.savefig(full_url)
                #     sns.plt.clf()   # Clear the figure to prepare for the next plot
                #     boxplots[compare_feat] = paths.EXAMPLES_RELATIVE + \
                #                                  str(base_feat + "_" + compare_feat + "_box.png")
                #     # z-test
                #     # t-test
                #     # ANOVA
                #     break
                # else:
                #     # Non-numeric and non-numeric
                #     # Stacked bar charts
                #
                #     # Chi squared
                #     # Crater's Chi
                #     # Mantel
                #     break

            # Create interaction object comparing this feature to all others
            interaction = Interaction(feat_name=base_feat,
                                      feat_index=feature_index,
                                      other_features=other_features,
                                      scatterplots=scatterplots,
                                      correlations=correlations,
                                      covariances=covariances,
                                      boxplots=boxplots,
                                      ztests=ztests,
                                      ttests=ttests,
                                      anova=anova,
                                      stackedbarplots=stackedbarplots,
                                      chisquared=chisquared,
                                      craters=craters,
                                      mantelhchi=mantelhchi)

            # Add to the collection of interactions
            interactions_collection[base_feat] = interaction
            feature_index += 1

        # Create interactions object to represent the entire collection
        interactions = Interactions(name=self.title,
                                    features=features,
                                    feature_interactions=interactions_collection)
        interactions_json = jsonpickle.encode(interactions)

        # # Get correlations between features
        # correlations = self.data.select_dtypes(include=['int64', 'float64']).corr()
        # correlation_heatmap = sns.heatmap(correlations, vmax=1, square=True)
        #
        # # Save the heatmap
        # correlation_url = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_corr.png"))
        # correlation_url_relative = paths.EXAMPLES_RELATIVE + str(self.title + "_corr.png")
        # fig = correlation_heatmap.get_figure()
        # fig.savefig(correlation_url)
        #
        # # Clear the figure to prepare for the next plot
        # sns.plt.clf()
        #
        # # Get covariance between features
        # covariance = self.data.cov()
        # covariance_heatmap = sns.heatmap(covariance, vmax=1, square=True)
        # covariance_url = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_cov.png"))
        # covariance_url_relative = paths.EXAMPLES_RELATIVE + str(self.title + "_cov.png")
        # fig = covariance_heatmap.get_figure()
        # fig.savefig(covariance_url)
        # sns.plt.clf()
        #
        # # Save data as JSON
        # interactions = Interactions(name=self.title,
        #                             correlations_url=correlation_url_relative,
        #                             covariance_url=covariance_url_relative)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + INTERACTIONS_SUFFIX)), 'w')
        file.write(interactions_json)
        file.close()

    def load_summary_json(self):
        return self.load_json(SUMMARY_SUFFIX)

    def load_features_json(self):
        return self.load_json(FEATURES_SUFFIX)

    def load_interactions_json(self):
        return self.load_json(INTERACTIONS_SUFFIX)

    def load_json(self, json_suffix):
        absolute_filename = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + json_suffix))

        # Check if the JSON file exists and if not, generate it
        if not os.path.isfile(absolute_filename):
            if json_suffix == SUMMARY_SUFFIX:
                self.generate_summary_json()
            elif json_suffix == FEATURES_SUFFIX:
                self.generate_features_json()
            elif json_suffix == INTERACTIONS_SUFFIX:
                self.generate_interactions_json()

        # Read serialized JSON file
        with open(absolute_filename, 'r') as serialized_file:
            json_str = serialized_file.read()
            deserialized_json = jsonpickle.decode(json_str)
        return deserialized_json



