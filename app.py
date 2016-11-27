from flask import Flask, render_template, session, request, redirect, url_for
from data_driver import DataDriver
from datasets import DataSets
from dataset import DataSet
import paths
import key
import os
import pandas as pd
import jsonpickle

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER
app.secret_key = key.SECRET_KEY

@app.before_first_request
def getmenu():
    if not os.path.isfile(paths.DATASETS_JSON):
        dataset_info = []

        # Read in CSV
        datasets_csv = pd.read_csv(paths.DATASETS)

        # Get title, filename, id, and label for each data set and add it to the collection
        for index, row in datasets_csv.iterrows():
            dataset_title = row["Title"]
            dataset_filename = row["FileName"]
            dataset_id = row["ID"]
            dataset_label = row["Label"]
            dataset = DataSet(dataset_filename, dataset_title, dataset_id, dataset_label)
            dataset_info.append(dataset)

        # Save the collection as JSON and return it
        datasets = DataSets(dataset_info=dataset_info)
        datasets_json = jsonpickle.encode(datasets)

        # Save the serialized JSON to a file
        with open(paths.DATASETS_JSON, 'w') as file:
            file.write(datasets_json)
    else:
         with open(paths.DATASETS_JSON, 'r') as serialized_file:
            json_str = serialized_file.read()
            datasets_json = jsonpickle.decode(json_str)

    return datasets_json


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

    if dataset is not None:
        new_title = dataset["Title"].values[0]
        new_index = dataset["ID"].values[0]
        new_label = dataset["Label"].values[0]

        # Save the selection in session
        session["data_file"] = new_selection
        session["data_title"] = new_title
        session["data_id"] = new_index
        session["data_label"] = new_label

    # Redirect and reload the appropriate page
    if request.referrer is not None:
        return redirect(request.referrer)
    else:
        return redirect(url_for('index'))


def datasetuploaded():
    data_file, data_title, data_id, data_label = selecteddataset()

    # Create folder with graphs subfolder
    data_path = os.path.join(paths.EXAMPLES_FOLDER, data_title)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        os.makedirs(os.path.join(data_path, "graphs"))

    # Save the uploaded file to this directory

    # Update the list of options to select from

@app.route('/')
@app.route('/index')
def index():
    data_file, data_title, data_id, data_label = selecteddataset()
    dataset_options = getmenu()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    summary_json = driver.load_summary_json()

    return render_template('index.html',
                           data=summary_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label,
                           dataset_options=dataset_options)


@app.route('/univariate')
def univariate():
    data_file, data_title, data_id, data_label = selecteddataset()
    dataset_options = getmenu()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    features_json = driver.load_features_json()

    return render_template('univariate.html',
                           mydata=features_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label,
                           dataset_options=dataset_options)


@app.route('/bivariate')
def bivariate():
    # Read Titanic data
    data_file, data_title, data_id, data_label = selecteddataset()
    dataset_options = getmenu()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    interactions_json = driver.load_interactions_json()

    return render_template('bivariate.html',
                           data=interactions_json,
                           data_file=data_file,
                           data_title=data_title,
                           data_id=data_id,
                           data_label=data_label,
                           dataset_options=dataset_options)
