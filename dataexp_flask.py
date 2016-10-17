import pandas as pd
import seaborn as sns
import os

class DataExplorer:

    def __init__(self, datafile, idcolumn, labelcolumn):
        self.data = pd.read_csv(datafile)
        self.id_column = idcolumn
        self.label_column = labelcolumn

        # Separate out the numeric fields (excluding ID)
        self.numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        self.numeric_fieldnames = list(self.numeric_data.columns.values)
        self.numeric_fieldnames.remove(self.id_column)

        self.APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLES_FOLDER = os.path.join(self.APP_ROOT, 'static/sampledata/')

    def get_stats(self):
        # Get count, num missing, num unique, avg, median, max, min, standard dev, and variance for each numeric field
        data_count = list(map(lambda x: self.numeric_data[x].count(), self.numeric_fieldnames))
        data_missing = list(map(lambda x: self.numeric_data[x].isnull().sum(), self.numeric_fieldnames))
        data_unique = list(map(lambda x: len(self.numeric_data[x].unique()), self.numeric_fieldnames))
        data_avg = list(map(lambda x: self.numeric_data[x].mean(), self.numeric_fieldnames))
        data_med = list(map(lambda x: self.numeric_data[x].median(), self.numeric_fieldnames))
        data_max = list(map(lambda x: self.numeric_data[x].max(), self.numeric_fieldnames))
        data_min = list(map(lambda x: self.numeric_data[x].min(), self.numeric_fieldnames))
        data_std = list(map(lambda x: self.numeric_data[x].std(), self.numeric_fieldnames))
        data_var = list(map(lambda x: self.numeric_data[x].var(), self.numeric_fieldnames))

        # Return statistics as a list
        stats = zip(self.numeric_fieldnames, data_count, data_missing, data_unique, data_avg, data_med, data_max,
                    data_min, data_std, data_var)
        return stats

    def get_headers(self):
        # Return the header for the stats list
        return ["Feature", "Count", "Missing", "Unique", "Average", "Median", "Max", "Min", "StdDev", "Variance"]

    def summary_plots(self):
        # Remove rows with nulls and show summary graphs for all numeric features
        features_only = self.numeric_fieldnames.copy()
        features_only.remove(self.label_column)
        return sns.pairplot(self.data.dropna(), hue=self.label_column, vars=features_only, diag_kind='kde')

    def get_histograms(self):
        hist = {}
        for feature in self.numeric_fieldnames:
            # Generate the histogram
            feature_data = self.data[feature].dropna()
            hist_plot = sns.distplot(feature_data, bins=None, hist=True, kde=False, rug=False)

            # Save the histogram
            full_url = os.path.join(self.EXAMPLES_FOLDER, str(feature + "_hist.png"))
            fig = hist_plot.get_figure()
            fig.savefig(full_url)

            # Clear the figure to prepare for the next plot
            sns.plt.clf()

            # Return the relative URL to the histogram
            hist[feature] = "static/sampledata/" + str(feature + "_hist.png")
        return hist