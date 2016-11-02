import pandas as pd
import seaborn as sns
import os
import paths

class DataExplorer:

    def __init__(self, datafile, idcolumn, labelcolumn, title):
        # CSV file, row id column name, label column name
        self.data = pd.read_csv(datafile)
        self.id_column = idcolumn
        self.label_column = labelcolumn
        self.title = title

        # Numeric fields (excluding ID)
        self.numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        self.numeric_fieldnames = list(self.numeric_data.columns.values)
        if self.id_column in self.numeric_fieldnames:
            self.numeric_fieldnames.remove(self.id_column)

        # Non-numeric fields (excluding ID)
        self.nonnumeric_data = self.data.select_dtypes(exclude=['int64', 'float64'])
        self.nonnumeric_fieldnames = list(self.nonnumeric_data.columns.values)
        if self.id_column in self.nonnumeric_fieldnames:
            self.nonnumeric_fieldnames.remove(self.id_column)

    def get_title(self):
        return self.title

    def get_stats_numeric(self):
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

    def get_stats_nonnumeric(self):
        data_count = list(map(lambda x: self.nonnumeric_data[x].count(), self.nonnumeric_fieldnames))
        data_missing = list(map(lambda x: self.nonnumeric_data[x].isnull().sum(), self.nonnumeric_fieldnames))
        data_unique = list(map(lambda x: len(self.nonnumeric_data[x].unique()), self.nonnumeric_fieldnames))
        data_mostcommon = list(map(lambda x: str("%s (%d)" % \
                            (self.nonnumeric_data[x].value_counts().idxmax(),
                             self.nonnumeric_data[x].value_counts().max())), self.nonnumeric_fieldnames))
        data_leastcommon = list(map(lambda x: str("%s (%d)" % \
                            (self.nonnumeric_data[x].value_counts().idxmin(),
                             self.nonnumeric_data[x].value_counts().min())), self.nonnumeric_fieldnames))

        stats = zip(self.nonnumeric_fieldnames, data_count, data_missing, data_unique, data_mostcommon,
                    data_leastcommon)

        return stats

    def find_most_common(self, column_data):
        # Get numeric categorical variables
        factors = pd.factorize(column_data)
        return factors.median()

    def get_headers_numeric(self):
        # Return the header for the stats list
        return ["Feature", "Count", "Missing", "Unique", "Average", "Median", "Max", "Min", "StdDev", "Variance"]

    def get_headers_nonnumeric(self):
        return ["Feature", "Count", "Missing", "Unique", "Most Common (#)", "Least Common (#)"]

    def get_histograms_numeric(self):
        hist = {}
        for feature in self.numeric_fieldnames:
            # Generate the histogram
            feature_data = self.data[feature].dropna()
            hist_plot = sns.distplot(feature_data, bins=None, hist=True, kde=False, rug=False)

            # Save the histogram
            full_url = os.path.join(paths.EXAMPLES_FOLDER, str(feature + "_hist.png"))
            fig = hist_plot.get_figure()
            fig.savefig(full_url)

            # Clear the figure to prepare for the next plot
            sns.plt.clf()

            # Return the relative URL to the histogram
            hist[feature] = paths.EXAMPLES_RELATIVE + str(feature + "_hist.png")
        return hist