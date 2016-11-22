from flask import Flask
from flask import render_template, session
from data_driver import DataDriver
import paths
import key
import jsonpickle

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER
app.secret_key = key.SECRET_KEY


@app.route('/')
@app.route('/index')
def index():
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    summary_json = driver.load_summary_json()

    return render_template('index.html', data=summary_json)


def datasetselected():
    return None


@app.route('/univariate')
def univariate():
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    features_json = driver.load_features_json()

    return render_template('univariate.html', mydata=features_json)


@app.route('/bivariate')
def bivariate():
    # Read Titanic data
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    interactions_json = driver.load_interactions_json()

    return render_template('bivariate.html', data=interactions_json)
