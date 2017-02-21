# Ugly hack to fix imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_bivariate import DataBivariate
from model.interactions import Interactions
from model.interaction import Interaction


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

    def test_get_feature_interactions_type(self):
        expected = Interaction
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = type(self.bivariate.get_feature_interactions("Survived", 0, features_list, stats_category_list))
        assert actual == expected

    def test_get_feature_interactions_int_statsbycategory(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.statsbycategory is not {}

    def test_get_feature_interactions_int_statsbycategoryflipped(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.statsbycategoryflipped is not {}

    def test_get_feature_interactions_int_frequencies(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.frequency_table is not {}

    def test_get_feature_interactions_int_boxplots(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.boxplots is not {}

    def test_get_feature_interactions_int_stackedbarplots(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.stackedbarplots is not {}

    def test_get_feature_interactions_int_correlations(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.correlations == {}

    def test_get_feature_interactions_int_covariance(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.covariances == {}

    def test_get_feature_interactions_int_chisquared(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.chisquared == {}

    def test_get_feature_interactions_int_cramers(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.cramers == {}

    def test_get_feature_interactions_int_scatterplots(self):
        features_list = self.bivariate.get_features_list()
        stats_category_list = self.bivariate.get_stats_by_category_list()
        actual = self.bivariate.get_feature_interactions("SibSp", 0, features_list, stats_category_list)
        assert actual.scatterplots == {}