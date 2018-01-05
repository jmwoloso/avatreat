import unittest
import pandas as pd

from avatreat.preprocessing import TreatmentDesign


class TestTreatmentDesign(unittest.TestCase):
    """Test suite for the avatreat.preprocessing.TreatmentDesign
    class."""
    def setUp(self):
        """Setup method for the test suite."""
        # test data
        data = {
            "i": [100, 101, 102, 103, 104],
            "t": [pd.datetime(2017, 6, 14) for i in range(5)],
            "u": [pd.np.nan, 1, 2, 3, 4],
            "v": [0, 1, 2, 3, 4],
            "w": ["a", "b", "a", "b", "a"],
            "x": ["a", "a", "b", "b", pd.np.nan],
            "y": [1, 1, 0, 1, 1]
        }

        # test dataframe
        self.df = pd.DataFrame(data)

    def test_fit_exclude_features(self):
        """Tests whether features are excluded correctly or not in
        the fit method."""
        ids = ["i"]
        datetimes = ["t"]
        target = "y"

        # instantiate the class
        td = TreatmentDesign(id_features=ids,
                             datetime_features=datetimes,
                             target=target)

        # fit to the test dataframe and ensure the appropriate
        # columns are excluded and that the last column is our target
        # column
        td.fit(self.df)

        # test cases
        last_feature="y"
        remaining_features=["u", "v", "w", "x", "y"]

        # test that the last feature is the target
        self.assertEqual(self.df.columns[-1], last_feature)
        self.assertListEqual(list(td.df_.columns), remaining_features)





    def tearDown(self):
        """Tear down method for the test suite."""
        del self.df