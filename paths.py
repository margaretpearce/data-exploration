import os
UPLOAD_RELATIVE = 'static/uploads/'
EXAMPLES_RELATIVE = 'static/data/'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_RELATIVE)
EXAMPLES_FOLDER = os.path.join(APP_ROOT, EXAMPLES_RELATIVE)
