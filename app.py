from flask import Flask
from flask import render_template
from dataexp_flask import DataExplorer
from data_driver import DataDriver
import paths
import key
import os

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
    driver.generate_summary_json() # TODO: Remove after testing
    summary_json = driver.load_summary_json()

    return render_template('index.html', data=summary_json)


def datasetselected():
    return None


@app.route('/features')
def features():
    datastats = DataExplorer(os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.csv"),
                             "PassengerId",
                             "Survived",
                             "Titanic (Kaggle)")

    data = {'name': datastats.get_title(),
            'headers': datastats.get_headers_numeric(),
            'headerscat': datastats.get_headers_nonnumeric(),
            'statslist': datastats.get_stats_numeric(),
            'statslistcat': datastats.get_stats_nonnumeric()
            }

    plot = {'url': os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.png")}

    hist_src = datastats.get_histograms_numeric()
    hist = {'hist_urls': hist_src}

    return render_template('features.html', data=data, plot=plot, hist=hist)


@app.route('/interactions')
def interactions():
    return render_template('interactions.html')
