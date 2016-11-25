from flask import Flask, render_template, session, request
from data_driver import DataDriver
import paths
import key
import os
import pandas as pd

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER
app.secret_key = key.SECRET_KEY


def selecteddataset():
    data_file = None
    data_title = None
    data_id = None
    data_label = None

    # Check if the values are already in session
    if "data_file" in session:
        data_file = session["data_file"]
    if "data_title" in session:
        data_title = session["data_title"]
    if "data_id" in session:
        data_id = session["data_id"]
    if "data_label" in session:
        data_label = session["data_label"]

    # Make sure that at least the file and title are populated, or else get it from the page
    if data_file is None or data_title is None:
        # Get the current selected values
        data_file = "iris.csv"
        data_title = "Iris"
        data_id = "ID"
        data_label = "Species"

        # Save values in session for future requests
        session["data_file"] = data_file
        session["data_title"] = data_title
        session["data_id"] = data_id
        session["data_label"] = data_label

    return data_file, data_title, data_id, data_label


@app.route('/dataset_selection_changed', methods=['POST'])
def dataset_selection_changed():
    # Get the selected data set's name
    new_selection = str(request.form["data_set_field"])

    # Look up the Title, ID, Label (for existing data sets)
    datasets = pd.read_csv(paths.DATASETS)
    dataset = datasets.loc[datasets["FileName"] == new_selection]
    new_title = dataset["Title"][1]
    new_index = dataset["ID"][1]
    new_label = dataset["Label"][1]

    # Save the selection in session
    session["data_file"] = new_selection
    session["data_title"] = new_title
    session["data_id"] = new_index
    session["data_label"] = new_label

    # Redirect and reload the appropriate page
    rule = request.url_rule

    if '/univariate' in rule.rule:
        return univariate()
    elif '/bivariate' in rule.rule:
        return bivariate()
    else:
        return index()


def datasetuploaded():
    data_file, data_title, data_id, data_label = selecteddataset()

    # Create folder with graphs subfolder
    data_path = os.path.join(paths.EXAMPLES_FOLDER, data_title)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        os.makedirs(os.path.join(data_path, "graphs"))

    # Save the uploaded file to this directory

@app.route('/')
@app.route('/index')
def index():
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    summary_json = driver.load_summary_json()

    return render_template('index.html',
                           data=summary_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label)


@app.route('/univariate')
def univariate():
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    features_json = driver.load_features_json()

    return render_template('univariate.html',
                           mydata=features_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label)


@app.route('/bivariate')
def bivariate():
    # Read Titanic data
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    interactions_json = driver.load_interactions_json()

    return render_template('bivariate.html',
                           data=interactions_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label)
