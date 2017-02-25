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
        cls.interaction_binary = cls.bivariate.get_feature_interactions("Survived", 0, cls.features_list,
                                                                       cls.stats_category_list)
        cls.interaction_string = cls.bivariate.get_feature_interactions("Embarked", 0, cls.features_list,
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

    # Base feat: binary
    def test_get_feature_interactions_binary_statsbycategory(self):
        actual = self.interaction_binary
        assert actual.statsbycategory is not {}

    def test_get_feature_interactions_binary_statsbycategoryflipped(self):
        actual = self.interaction_binary
        assert actual.statsbycategoryflipped == {}

    def test_get_feature_interactions_binary_frequencies(self):
        actual = self.interaction_binary
        assert actual.frequency_table is not {}

    def test_get_feature_interactions_binary_boxplots(self):
        actual = self.interaction_binary
        assert actual.boxplots is not {}

    def test_get_feature_interactions_binary_stackedbarplots(self):
        actual = self.interaction_binary
        assert actual.stackedbarplots is not {}

    def test_get_feature_interactions_binary_correlations(self):
        actual = self.interaction_binary
        assert actual.correlations == {}

    def test_get_feature_interactions_binary_covariance(self):
        actual = self.interaction_binary
        assert actual.covariances == {}

    def test_get_feature_interactions_binary_chisquared(self):
        actual = self.interaction_binary
        assert actual.chisquared is not {}

    def test_get_feature_interactions_binary_cramers(self):
        actual = self.interaction_binary
        assert actual.cramers is not {}

    def test_get_feature_interactions_binary_scatterplots(self):
        actual = self.interaction_binary
        assert actual.scatterplots == {}

    # Base feat: string
    def test_get_feature_interactions_string_statsbycategory(self):
        actual = self.interaction_string
        assert actual.statsbycategory is not {}

    def test_get_feature_interactions_string_statsbycategoryflipped(self):
        actual = self.interaction_string
        assert actual.statsbycategoryflipped == {}

    def test_get_feature_interactions_string_frequencies(self):
        actual = self.interaction_string
        assert actual.frequency_table is not {}

    def test_get_feature_interactions_string_boxplots(self):
        actual = self.interaction_string
        assert actual.boxplots is not {}

    def test_get_feature_interactions_string_stackedbarplots(self):
        actual = self.interaction_string
        assert actual.stackedbarplots is not {}

    def test_get_feature_interactions_string_correlations(self):
        actual = self.interaction_string
        assert actual.correlations == {}

    def test_get_feature_interactions_string_covariance(self):
        actual = self.interaction_string
        assert actual.covariances == {}

    def test_get_feature_interactions_string_chisquared(self):
        actual = self.interaction_string
        assert actual.chisquared is not {}

    def test_get_feature_interactions_string_cramers(self):
        actual = self.interaction_string
        assert actual.cramers is not {}

    def test_get_feature_interactions_string_scatterplots(self):
        actual = self.interaction_string
        assert actual.scatterplots == {}

    def test_get_correlation_lower_bound(self):
        correlation = self.bivariate.get_correlation("Age", "Fare")
        assert -1 <= correlation

    def test_get_correlation_upper_bound(self):
        correlation = self.bivariate.get_correlation("Age", "Fare")
        assert correlation <= 1

    def test_get_correlation_one_noncontinuous(self):
        correlation = self.bivariate.get_correlation("Age", "Name")
        assert correlation is None

    def test_get_correlation_both_noncontinuous(self):
        correlation = self.bivariate.get_correlation("Embarked", "Name")
        assert correlation is None

    def test_chisquared_pvalue_lower_bound(self):
        chisquared = self.bivariate.get_chisquared("Survived", "Sex")
        p_value = chisquared[1]
        assert 0 <= p_value

    def test_chisquared_pvalue_upper_bound(self):
        chisquared = self.bivariate.get_chisquared("Survived", "Sex")
        p_value = chisquared[1]
        assert p_value <= 1

    def test_chisquared_pvalue_one_continuous(self):
        chisquared = self.bivariate.get_chisquared("Survived", "Age")
        assert chisquared is None

    def test_chisquared_pvalue_both_continuous(self):
        chisquared = self.bivariate.get_chisquared("Fare", "Age")
        assert chisquared is None

    def test_cramersv_range(self):
        cramersv = self.bivariate.get_cramersv("Survived", "Pclass")
        assert 0 <= cramersv and cramersv <= 1

    def test_cramersv_one_continuous(self):
        cramersv = self.bivariate.get_cramersv("Survived", "Age")
        assert cramersv is None

    def test_cramersv_both_continuous(self):
        cramersv = self.bivariate.get_cramersv("Fare", "Age")
        assert cramersv is None
