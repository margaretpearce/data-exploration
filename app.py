from flask import Flask
from flask import render_template
from data_driver import DataDriver
import paths
import key

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER
app.secret_key = key.SECRET_KEY


@app.route('/')
@app.route('/index')
def index():
    # Read Titanic data
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    driver.generate_summary_json()  # TODO: Remove after testing
    summary_json = driver.load_summary_json()

    return render_template('index.html', data=summary_json)


def datasetselected():
    return None


@app.route('/univariate')
def univariate():
    # Read Titanic data
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    driver.generate_features_json()  # TODO: Remove after testing
    features_json = driver.load_features_json()

    return render_template('univariate.html', mydata=features_json)


@app.route('/bivariate')
def bivariate():
    # Read Titanic data
    driver = DataDriver("titanic.csv", "Titanic", "PassengerId", "Survived")

    # Get the JSON for the summary data
    driver.generate_interactions_json()  # TODO: Remove after testing
    interactions_json = driver.load_interactions_json()

    return render_template('bivariate.html', data=interactions_json)
