import os
UPLOAD_RELATIVE = 'static/uploads/'
EXAMPLES_RELATIVE = 'static/data/'
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_RELATIVE)
EXAMPLES_FOLDER = os.path.join(APP_ROOT, EXAMPLES_RELATIVE)
GRAPHS_SUBFOLDER = "/graphs/"

DATASETS = os.path.join(EXAMPLES_FOLDER, "datasets.csv")
DATASETS_JSON = os.path.join(EXAMPLES_FOLDER, "datasets.json")

# Graph types
FILE_BARCHART = "_bar.png"
FILE_BOXCHART = "_box.png"
FILE_SCATTERPLOT = "_scatter.png"
FILE_COUNTPLOT = "_countplot.png"
FILE_HISTOGRAM = "_hist.png"
