from flask import Flask
from flask import render_template
from dataexp_flask import DataExplorer
import os
import seaborn as sns
import base64

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
EXAMPLES_FOLDER = os.path.join(APP_ROOT, 'static/sampledata/')
app.config['EXAMPLES_FOLDER'] = EXAMPLES_FOLDER

@app.route('/')
@app.route('/index')
def index():
    datastats = DataExplorer(os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.csv"), "PassengerId", "Survived")

    data = {'name': "Titanic (Kaggle Training Set)",
            'headers' : datastats.get_headers(),
            'statslist' : datastats.get_stats()}

    plot_src = datastats.summary_plots()
    plot_url = os.path.join(app.config['EXAMPLES_FOLDER'], "titanic.png")
    plot_src.savefig(plot_url)
    plot = {'url' : "static/sampledata/titanic.png"}

    return render_template('index.html', data=data, plot=plot)