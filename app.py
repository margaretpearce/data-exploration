from flask import Flask
from flask import render_template, session
from data_driver import DataDriver
import paths
import key
import json, jsonpickle

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER
app.secret_key = key.SECRET_KEY


@app.route('/')
@app.route('/index')
def index():
    if 'summary_json' in session:
        summary_string = session['summary_json']
        summary_json = jsonpickle.decode(summary_string)
    else:
        # Read Titanic data
        driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

        # Get the JSON for the summary data
        driver.generate_summary_json()  # TODO: Remove after testing
        summary_json = driver.load_summary_json()

        # Save the JSON encoded string in session
        session['summary_json'] = jsonpickle.encode(summary_json)

    return render_template('index.html', data=summary_json)


def datasetselected():
    return None


@app.route('/univariate')
def univariate():
    if 'features_json' in session:
        features_string = session['features_json']
        features_json = jsonpickle.decode(features_string)
        print("loaded features_json")
    else:
        # Read Titanic data
        driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

        # Get the JSON for the summary data
        driver.generate_features_json()  # TODO: Remove after testing
        features_json = driver.load_features_json()

        # Save the JSON encoded string in session
        session['features_json'] = jsonpickle.encode(features_json)

    return render_template('univariate.html', mydata=features_json)


@app.route('/bivariate')
def bivariate():
    if 'interactions_json' in session:
        interactions_string = session['interactions_json']
        interactions_json = jsonpickle.decode(interactions_string)
        print("loaded interactions_json")
    else:
        # Read Titanic data
        driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

        # Get the JSON for the summary data
        driver.generate_interactions_json()  # TODO: Remove after testing
        interactions_json = driver.load_interactions_json()

        # Save the JSON encoded string in session
        session['interactions_json'] = jsonpickle.encode(interactions_json)
        print("saving interactions in session")

    return render_template('bivariate.html', data=interactions_json)
