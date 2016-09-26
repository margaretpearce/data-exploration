import pandas as pd
import seaborn as sns

class DataExplorer:
    def __init__(self, datafile, idcolumn, labelcolumn):
        self.data = pd.read_csv(datafile)
        self.id_column = idcolumn
        self.label_column = labelcolumn
        self.numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        self.numeric_fieldnames = list(self.numeric_data.columns.values)
        self.numeric_fieldnames.remove(self.id_column)

    def get_stats(self):
        data_count = list(map(lambda x: self.numeric_data[x].count(), self.numeric_fieldnames))
        data_avg = list(map(lambda x: self.numeric_data[x].mean(), self.numeric_fieldnames))
        data_med = list(map(lambda x: self.numeric_data[x].median(), self.numeric_fieldnames))
        data_max = list(map(lambda x: self.numeric_data[x].max(), self.numeric_fieldnames))
        data_min = list(map(lambda x: self.numeric_data[x].min(), self.numeric_fieldnames))
        data_std = list(map(lambda x: self.numeric_data[x].std(), self.numeric_fieldnames))
        data_var = list(map(lambda x: self.numeric_data[x].var(), self.numeric_fieldnames))

        stats = zip(self.numeric_fieldnames, data_count, data_avg, data_med, data_max, data_min, data_std, data_var)
        return stats

    def get_headers(self):
        return ["Feature", "Count", "Average", "Median", "Max", "Min", "StdDev", "Variance"]

    def summary_plots(self):
        # Remove rows with nulls and show summary graphs
        features_only = self.numeric_fieldnames.copy()
        features_only.remove(self.label_column)
        return sns.pairplot(self.data.dropna(), hue=self.label_column, vars=features_only, diag_kind='kde')