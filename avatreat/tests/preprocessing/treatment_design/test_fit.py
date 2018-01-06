import unittest
from avatreat.utils.routines.fit import *


class TestTreatmentDesign(unittest.TestCase):
    """Test suite for the `fit` method in
    avatreat.preprocessing.TreatmentDesign class."""
    def setUp(self):
        """Setup method for the test suite."""
        # test data
        #TODO [TST]: shore these up to make use of the same column for multiple tests/functions
        data = {
            # non-treatment features
            "index": ["100A", "101A", "102A", "103A", "104A"],
            "datetime": [pd.datetime(2017, 6, 14) for i in range(5)],
            # target feature
            "target": [1, 1, 0, 1, 1],
            # ints
            "n": [pd.np.nan, 1, 2, 3, 4],
            "nn": ["1", "2", "3", "4", ""],
            "nnn": ["1", "2", "3", "4", pd.np.nan],
            # floats
            "f": [0.0, 1.1, 2.2, 3.3, 4.4],
            "ff": [0.0, 1.1, 2.2, 3.3, pd.np.nan],
            "fff": ["0.0", "1.1", "2.2", "3.3", "4.4"],
            # booleans
            "i": [True, False, True, False, True],
            "ii": ["true", "false", "true", "false", "true"],
            # categoricals
            "c": ["a", "b", "a", "b", "a"],
            "cc": ["a", "a", "b", "b", pd.np.nan],
            "ccc": ["a", "b", "c", "d", "e"],  # high-cardinality
            # zero-variance features
            "z": ["a", "a", "a", "a", "a"],
            "zz": [0, 0, 0, 0, 0],
            "zzz": [0.0, 0.0, 0.0, 0.0, 0.0]
        }

        # test dataframe
        self.df = pd.DataFrame(data)


    def test_find_hidden_dtypes(self):
        """Tests functionality of `find_hidden_dtypes`."""

        true_numericals = ["f", "ff", "fff", "n", "nn", "nnn",
                           "target", "zz", "zzz"]

        true_booleans = ["i", "ii"]

        df = find_hidden_dtypes(dataframe=self.df)

        numerical_features = df.select_dtypes(
            include=NUMERICAL_DTYPES).columns.tolist()

        numerical_features.sort()

        boolean_features = df.select_dtypes(include=[
            "bool_"]).columns.tolist()

        boolean_features.sort()

        self.assertListEqual(numerical_features,
                             true_numericals,
                             "numerical features do not match.")

        self.assertListEqual(boolean_features,
                             true_booleans,
                             "boolean features do not match.")


    def test_fill_missing_values(self):
        """Tests functionality of `fill_missing_values`."""
        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        err_msg = "values do not match."

        self.assertEqual(df.loc[0, "n"], -1.0, err_msg)
        self.assertEqual(df.loc[4, "nnn"], -1.0, err_msg)
        self.assertEqual(df.loc[4, "ff"], -1.0, err_msg)
        self.assertEqual(df.loc[4, "cc"], "NA", err_msg)


    def test_find_zero_variance_features(self):
        """Tests functionality of `find_zero_variance_features`."""
        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        blacklist_ = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=True
        )

        blacklist_.sort()

        # this should return an empty list
        blacklist = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=False
        )

        # ground truth
        true_blacklist = ["z", "zz", "zzz"]
        false_blacklist = list()

        err_msg = "lists do not match."

        self.assertListEqual(blacklist_,
                             true_blacklist,
                             err_msg)

        self.assertEqual(blacklist,
                         false_blacklist,
                         err_msg)


    def test_get_treatment_features(self):
        """Tests that the correct features are returned from
        `get_treatment_features`"""
        index = ["index"]
        datetimes = ["datetime"]
        target = "target"

        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        blacklist_ = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=True
        )

        # used in one or more tests
        treatment_features_ = get_treatment_features(
            dataframe=df,
            id_features=index,
            datetime_features=datetimes,
            target=target,
            blacklist=blacklist_)

        treatment_features_.sort()

        # ground truth
        remaining_features=["c", "cc", "ccc", "f", "ff", "fff", "i",
                            "ii", "n", "nn", "nnn"]

        err_msg = "lists are not equal."

        # test that the remaining features are correct
        self.assertListEqual(treatment_features_,
                             remaining_features)


    def test_reindex_target(self):
        """Tests that the target feature gets moved to the end of the
        dataframe."""
        target = "target"

        df = reindex_target(dataframe=self.df,
                            target=target)

        # ground truth
        last_feature="target"

        err_msg = "last feature is not the target."
        # test that the last feature is the target
        self.assertEqual(df.columns[-1], last_feature, err_msg)


    def test_cast_to_int(self):
        """Tests the functionality of `cast_to_int`."""
        index = ["index"]
        datetimes = ["datetime"]
        target = "target"

        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        blacklist_ = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=True
        )

        # used in one or more tests
        treatment_features_ = get_treatment_features(
            dataframe=df,
            id_features=index,
            datetime_features=datetimes,
            target=target,
            blacklist=blacklist_)
        # print(treatment_features_)
        df = reindex_target(dataframe=df,
                            target=target)


        df = cast_to_int(dataframe=df,
                         treatment_features=treatment_features_)

        ints = df.loc[:, treatment_features_]\
            .select_dtypes(include=INT_DTYPES).columns.tolist()
        ints.sort()

        # ground truth
        true_ints = ["n", "nn", "nnn"]

        err_msg = "lists are not equal."

        self.assertListEqual(ints, true_ints, err_msg)


    def test_get_column_dtypes(self):
        """Tests the functionality of `get_column_dtypes`."""
        index = ["index"]
        datetimes = ["datetime"]
        target = "target"

        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        blacklist_ = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=True
        )

        # used in one or more tests
        treatment_features_ = get_treatment_features(
            dataframe=df,
            id_features=index,
            datetime_features=datetimes,
            target=target,
            blacklist=blacklist_)

        df = reindex_target(dataframe=df,
                            target=target)

        df = cast_to_int(dataframe=df,
                         treatment_features=treatment_features_)

        ints, floats, objs, bools = get_column_dtypes(
            dataframe=df,
            treatment_features=treatment_features_)

        ints.sort()
        floats.sort()
        objs.sort()
        bools.sort()

        # ground truth
        true_ints = ["n", "nn", "nnn"]
        true_floats = ["f", "ff", "fff"]
        true_objs = ["c", "cc", "ccc"]
        true_bools = ["i", "ii"]

        self.assertListEqual(ints, true_ints, "int lists do not match.")
        self.assertListEqual(floats, true_floats, "float lists do not match.")
        self.assertListEqual(objs, true_objs, "obj lists do not match.")
        self.assertListEqual(bools, true_bools, "bool lists do not match.")


    def test_high_cardinality_features(self):
        """Tests that object columns are divided correctly between
        high-cardinality and categorical features."""
        index = ["index"]
        datetimes = ["datetime"]
        target = "target"

        df = find_hidden_dtypes(dataframe=self.df)

        df = fill_missing_values(dataframe=df,
                                 numerical_fill_value=-1.0,
                                 categorical_fill_value="NA")

        blacklist_ = find_zero_variance_features(
            dataframe=df,
            exclude_zero_variance_features=True
        )

        # used in one or more tests
        treatment_features_ = get_treatment_features(
            dataframe=df,
            id_features=index,
            datetime_features=datetimes,
            target=target,
            blacklist=blacklist_)

        df = reindex_target(dataframe=df,
                            target=target)

        df = cast_to_int(dataframe=df,
                         treatment_features=treatment_features_)

        ints, floats, objs, bools = get_column_dtypes(
            dataframe=df,
            treatment_features=treatment_features_)

        hc_features, cat_features = find_high_cardinality_features(
            dataframe=df,
            object_features=objs,
            rare_level_threshold=0.2,
            allowable_rare_percentage=0.2)

        # test cases
        true_hc_features = ["ccc"]
        true_cat_features = ["c", "cc"]

        self.assertListEqual(hc_features,
                             true_hc_features,
                             "high-cardinality features do not match.")
        self.assertListEqual(cat_features,
                             true_cat_features,
                             "categorical features do not match.")


    def tearDown(self):
        """Tear down method for the test suite."""
        del self.df