from flask import Flask
from flask import render_template
from dataexp_flask import DataExplorer
import paths
import os

app = Flask(__name__)

# Path information
app.config['UPLOAD_FOLDER'] = paths.UPLOAD_FOLDER
app.config['EXAMPLES_FOLDER'] = paths.EXAMPLES_FOLDER

@app.route('/')
@app.route('/index')
def index():
    datastats = DataExplorer(os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.csv"),
                             "PassengerId",
                             "Survived",
                             "Titanic (Kaggle)")

    data = {'name': datastats.get_title(),
            'headers' : datastats.get_headers_numeric(),
            'headerscat': datastats.get_headers_nonnumeric(),
            'statslist' : datastats.get_stats_numeric(),
            'statslistcat' : datastats.get_stats_nonnumeric()
            }

    plot = {'url' : os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.png")}

    hist_src = datastats.get_histograms_numeric()
    hist = {'hist_urls' : hist_src}

    return render_template('index.html', data=data, plot=plot, hist=hist)

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/interactions')
def interactions():
    return render_template('interactions.html')