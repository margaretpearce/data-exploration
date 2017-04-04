Data Exploration Tool

Reads in a csv, tsv, or Excel file with optionally specified index and label field, then presents summary statistics on the data set in addition to bivariate and univariate analysis for each variable/ pair of variables.

See a running demo at: http://adegene.com/dataexplorer


Dependencies:
- pandas
- numpy
- seaborn
- scipy
- jsonpickle
- flask
- xlrd
- Python 3.5 (other versions are untested as of now)


Supported data formats: 
- csv
- tsv
- xls
- xlsx


To run locally:

- Clone the repository
- Create a file keys.py under /configuration with a single variable: SECRET_KEY = '' (insert a random secret key - see Flask documentation)
- export FLASK_APP=app.py
- python -m flask run
