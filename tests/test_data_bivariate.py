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
        cls.features_list = cls.bivariate.get_features_list()
        cls.stats_category_list = cls.bivariate.get_stats_by_category_list()
        cls.interaction_int = cls.bivariate.get_feature_interactions("SibSp", 0, cls.features_list,
                                                                     cls.stats_category_list)
        cls.interaction_float = cls.bivariate.get_feature_interactions("Age", 0, cls.features_list,
                                                                       cls.stats_category_list)

    def test_load_bivariate_json(self):
        interactions_loaded = self.bivariate.load_interactions_json()
        actual = type(interactions_loaded)
        expected = Interactions
        assert actual == expected

    def test_get_feature_interactions_type(self):
        expected = Interaction
        actual = type(self.interaction_int)
        assert actual == expected

    # Base feat: integer
    def test_get_feature_interactions_int_statsbycategory(self):
        actual = self.interaction_int
        assert actual.statsbycategory is not {}

    def test_get_feature_interactions_int_statsbycategoryflipped(self):
        actual = self.interaction_int
        assert actual.statsbycategoryflipped is not {}

    def test_get_feature_interactions_int_frequencies(self):
        actual = self.interaction_int
        assert actual.frequency_table is not {}

    def test_get_feature_interactions_int_boxplots(self):
        actual = self.interaction_int
        assert actual.boxplots is not {}

    def test_get_feature_interactions_int_stackedbarplots(self):
        actual = self.interaction_int
        assert actual.stackedbarplots is not {}

    def test_get_feature_interactions_int_correlations(self):
        actual = self.interaction_int
        assert actual.correlations == {}

    def test_get_feature_interactions_int_covariance(self):
        actual = self.interaction_int
        assert actual.covariances == {}

    def test_get_feature_interactions_int_chisquared(self):
        actual = self.interaction_int
        assert actual.chisquared == {}

    def test_get_feature_interactions_int_cramers(self):
        actual = self.interaction_int
        assert actual.cramers == {}

    def test_get_feature_interactions_int_scatterplots(self):
        actual = self.interaction_int
        assert actual.scatterplots == {}

    # Base feat: float
    def test_get_feature_interactions_float_statsbycategory(self):
        actual = self.interaction_float
        assert actual.statsbycategory == {}

    def test_get_feature_interactions_float_statsbycategoryflipped(self):
        actual = self.interaction_float
        assert actual.statsbycategoryflipped is not {}

    def test_get_feature_interactions_float_frequencies(self):
        actual = self.interaction_float
        assert actual.frequency_table == {}

    def test_get_feature_interactions_float_boxplots(self):
        actual = self.interaction_float
        assert actual.boxplots is not {}

    def test_get_feature_interactions_float_stackedbarplots(self):
        actual = self.interaction_float
        assert actual.stackedbarplots == {}

    def test_get_feature_interactions_float_correlations(self):
        actual = self.interaction_float
        assert actual.correlations is not {}

    def test_get_feature_interactions_float_covariance(self):
        actual = self.interaction_float
        assert actual.covariances is not {}

    def test_get_feature_interactions_float_chisquared(self):
        actual = self.interaction_float
        assert actual.chisquared is not {}

    def test_get_feature_interactions_float_cramers(self):
        actual = self.interaction_float
        assert actual.cramers is not {}

    def test_get_feature_interactions_float_scatterplots(self):
        actual = self.interaction_float
        assert actual.scatterplots is not {}
