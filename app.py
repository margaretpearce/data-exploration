from flask import Flask
from flask import render_template, session
from data_driver import DataDriver
import paths
import key

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
        data_file = "titanic.csv"
        data_title = "Titanic"
        data_id = "PassengerId"
        data_label = "Survived"

        # Save values in session for future requests
        session["data_file"] = data_file
        session["data_title"] = data_title
        session["data_id"] = data_id
        session["data_label"] = data_label

    return data_file, data_title, data_id, data_label


@app.route('/')
@app.route('/index')
def index():
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    summary_json = driver.load_summary_json()

    return render_template('index.html', data=summary_json)


def dataset_selection_changed():
    # Get the dropdown values and save them in session
    return None


@app.route('/univariate')
def univariate():
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    features_json = driver.load_features_json()

    return render_template('univariate.html', mydata=features_json)


@app.route('/bivariate')
def bivariate():
    # Read Titanic data
    data_file, data_title, data_id, data_label = selecteddataset()
    driver = DataDriver(data_file, data_title, data_id, data_label)

    # Get the JSON for the summary data
    interactions_json = driver.load_interactions_json()

    return render_template('bivariate.html', data=interactions_json)
