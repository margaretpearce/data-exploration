# Ugly hack to fix imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_bivariate import DataBivariate
from model.interactions import Interactions


class TestDataBivariate:
    @classmethod
    def setup_class(cls):
        cls.dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
        cls.bivariate = DataBivariate(cls.dataset)
        cls.bivariate.load_data()

    def test_load_bivariate_json(self):
        interactions_loaded = self.bivariate.load_interactions_json()
        actual = type(interactions_loaded)
        expected = Interactions
        assert actual == expected