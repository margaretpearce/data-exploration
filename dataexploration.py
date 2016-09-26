import pandas as pd
import seaborn as sns
import sys

# Read in the data source
datafile = sys.argv[1]
data = pd.read_csv(datafile)

# Find the columns that have numeric data
numeric_data = data.select_dtypes(include=['int64', 'float64'])
numeric_fieldnames = list(numeric_data.columns.values)

# Get summary statistics for each numeric column
data_count = list(map(lambda x: numeric_data[x].count(), numeric_fieldnames))
data_avg = list(map(lambda x: numeric_data[x].mean(), numeric_fieldnames))
data_med = list(map(lambda x: numeric_data[x].median(), numeric_fieldnames))
data_max = list(map(lambda x: numeric_data[x].max(), numeric_fieldnames))
data_min = list(map(lambda x: numeric_data[x].min(), numeric_fieldnames))
data_std = list(map(lambda x: numeric_data[x].std(), numeric_fieldnames))
data_var = list(map(lambda x: numeric_data[x].var(), numeric_fieldnames))

# Print statistics in evenly spaced table
print ("%20s %10s %10s %10s %10s %10s %10s %10s" %
       ("Feature", "Count", "Average", "Median", "Max", "Min", "StdDev", "Variance"))
for i in range(len(numeric_fieldnames)):
    print ("%20s %10.0f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" %
           (numeric_fieldnames[i], data_count[i], data_avg[i], data_med[i],
            data_max[i], data_min[i], data_std[i], data_var[i]))

# Remove rows with nulls and show summary graphs
sns.pairplot(data.dropna()[numeric_fieldnames])
sns.plt.show()