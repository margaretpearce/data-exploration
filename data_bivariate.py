import os
import jsonpickle
import seaborn as sns

from data_driver import DataDriver
from model.feature import Feature
from model.features import Features
from configuration import paths
from configuration import const_types


class DataBivariate(DataDriver):
    def __init__(self, selected_dataset):
        DataDriver.__init__(self, selected_dataset)

