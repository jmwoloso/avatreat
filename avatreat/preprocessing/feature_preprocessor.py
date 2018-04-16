import numpy as np
import pandas as pd

from avatreat.utils.constants import OBJECT_DTYPES, INT_DTYPES, \
    FLOAT_DTYPES, NUMERICAL_DTYPES, DATETIME_DTYPES, TIMEDELTAS, \
    CATEGORICAL_DTYPES, DATETIMETZ_DTYPES, BOOL_DTYPES

from avatreat.utils.treatment_design import \
    cast_to_int, \
    find_high_cardinality_features


class FeaturePreprocessor(object):
    def __init__(self, index_feature=None,
                 target=None, target_type="categorical",
                 find_hidden_numerics=False,
                 exclude_zero_variance_features=True,
                 floats_to_ints=False,
                 ints_as_categories=True,
                 missing_numeric_strategy="systematically",
                 numerical_fill_value=-1.0,
                 categorical_fill_value="NA", rare_level_threshold=0.02,
                 allowable_rare_percentage=0.1,
                 variable_significance_threshold="1/n",
                 smoothing_factor=0.0,
                 dtype_compression=False):
        """
        Class for designing data treatments.

        Parameters
        ----------
        index_feature:   str; default=None; name of the feature that
        serves as an identifying index for the dataset. this feature
        will be excluded from treatment design.

        target:   one of {str, None}; default=None; name of the
        target feature in the dataset.

        target_type:   one of {"categorical", "numerical"};
        default="categorical"; whether the design is for a regression or
        classification target.

        find_hidden_numerics: bool; default=False; whether to try and
        find numerical features among any features that identify as the
        object dtype.

        exclude_zero_variance_features: bool; default=True; whether
        to eliminate features with no variance from treatment design
        as they contain no useful information.

        floats_to_ints: bool; default=False; whether to attempt to
        cast floats to ints, with the reasoning that floats which can
        successfully be cast to ints are probably really ordinal or
        categorical features masquerading as floats.

        ints_as_categories:   bool; default=False; whether to
        attempt converting numerical features with dtype=int to
        categorical features; if True, parameters relating to
        categorical features (`rare_level_threshold`,
        `allowable_rare_percentage`) will be similarly applied.

        NOTE: both `floats_to_ints` and `ints_as_categories` make
        aggressive assumptions hence the default values of
        False for both.

        missing_numeric_strategy:   one of {"systematically",
        "random"}; default="systematically"; the strategy to employ
        when filling in missing numerical values; used in conjunction
        with `numerical_fill_value`; if strategy is set to "random" then
        `numerical_fill_value` will be set to "mean" automatically
        regardless of the value provided.

        NOTE: the default strategy of "systematically" is the safe
        choice as importance is often implicit in missing values.

        numerical_fill_value:   one of {float, "mean"};
        default=-1.0; value to fill in missing numerical rows with;
        used in conjunction with `missing_numerical_strategy`; will
        be converted to np.int when attempting to fill missing int
        values.



        categorical_fill_value:   str; default="NA"; string to be
        used to fill in missing categorical levels.

        rare_level_threshold:   float; default=0.02; the
        percetage of the dataset that a categorical level must not
        exceed in order to be classified as (and treated accordingly) a
        rare level.

        NOTE: rare levels will be pooled together when creating
        indicators for categorical features.

        allowable_rare_percentage:   float; default=0.1; total
        percentage of the dataset that all "rare" levels of a
        categorical feature must not exceed in order to pool rare
        levels; aims to prevent categorical features with
        high-cardinality (many levels) from having rare levels pooled
        together since most levels are rare in that case; features
        exceeding this level will be classified as having
        high-cardinality and will not have indicators created.

        variable_significance_threshold:   one of {"1/n", float,
        None}; default="1/n" features; significance level that a
        feature must achieve in order to remain in the final treated
        dataset; if `None` all features will be kept.

        smoothing_factor:   float; default=0.0; number of
        pseudo-observations to add as a Laplace smoothing factor;
        reduces the range of predictions of rare levels.

        dtype_compression:   bool; default=True; whether to
        downcast column dtypes (e.g. np.int64 -> np.int8);
        downcasting depends on the range of values present for a
        given column and may not always be possible/successful;
        useful for very large datasets as it helps reduce the
        in-memory footprint of the data.

        Attributes
        ----------
        df_:   copy of the supplied dataframe used for modification

        excluded_features_: features to be excluded from treatment
        design but not removed from the df_ attribute;

        removed_features_: features that have been removed from the
        df_ attribute.

        treatment_features_:   features to be used for treatment design.

        object_features_:   remaining features with the `object` dtype.

        categorical_features_:  remaining features with the `category`
        dtype.

        boolean_features_:  features with the `bool_` dtype.

        integer_features_:  features with the `integer` dtype.

        float_features_:    features with the `float` dtype.

        datetime_features_: features with the `datetime` dtype.

        datetime_timezone_features_:    features with the datetime
        timezone dtype.

        timedelta_features_:    features with the timedelta dtype.

        high_cardinality_features_: features considered to be
        high-cardinality features (high relative number of category
        levels).

        categorical_features_:  features that can be safely converted to
        the categorical dtype.


        Returns
        -------
        self    :   object

        """
        # if no id features are supplied, replace None with empty list
        self.index_feature = index_feature
        self.target = target
        self.target_type = target_type
        self.find_hidden_numerics = find_hidden_numerics
        self.exclude_zero_variance_features = exclude_zero_variance_features
        self.floats_to_ints = floats_to_ints
        self.ints_as_categories = ints_as_categories
        self.missing_numerical_strategy = missing_numeric_strategy
        self.numerical_fill_value = numerical_fill_value if \
            missing_numeric_strategy == "systematically" else "mean"

        self.categorical_fill_value = categorical_fill_value
        self.rare_level_threshold = rare_level_threshold
        self.allowable_rare_percentage = allowable_rare_percentage
        self.variable_significance_threshold = variable_significance_threshold
        self.smoothing_factor = smoothing_factor
        self.dtype_compression = dtype_compression



    def fit(self, X):
        """
        Runs the treatment design processes on the dataset.

        Parameters
        ----------
        X   :   pandas.DataFrame instance; dataframe used to to
        design treatments.

        """
        self.is_fitted_ = False
        # copy the dataframe for modification
        self.df_ = X.copy()

        self.excluded_features_ = list()

        if self.target is not None:
            self.excluded_features_.append(self.target)
        if self.index_feature is not None:
            self.excluded_features_.append(self.index_feature)

        # get the available dtypes
        self.object_features_, self.categorical_features_, \
        self.boolean_features_, self.integer_features_, \
        self.float_features_, self.datetime_features_, \
        self.datetime_timezone_features_, self.timedelta_features_ = \
            self._get_dtypes()

        # find any hidden numerics within the object dtypes
        if self.find_hidden_numerics is True:
            self.hidden_ints_, self.hidden_floats_ = \
                self._find_hidden_numerics()
        else:
            self.hidden_ints_ = list()
            self.hidden_floats_ = list()



        # TODO: add support for feature engineering with datetimes
        # and then remove this method
        self.removable_features_ = self._get_removable_features()

        # find features with missing values

        # find features with zero variance
        if self.exclude_zero_variance_features is True:
            self.zero_variance_features_ = \
                self._find_zero_variance_features()
        else:
            self.zero_variance_features_ = list()

        # get a list of the features to be included in treatment
        # design
        self.treatment_features_ = \
            self._get_treatment_features()

        # find features with missing values
        self.features_with_nans_ = self._find_nan_features()

        # attempt to cast floats to ints
        if self.floats_to_ints is True:
            # try and cast float features to ints
            self.integer_castables_ = self._cast_to_int()
        else:
            self.integer_castables_ = list()

















        # # find high-cardinality features from within object features
        # self.high_cardinality_features_, self.categorical_features_ = \
        #     find_high_cardinality_features(dataframe=self.df_,
        #                                    object_features=self.object_features_,
        #                                    rare_level_threshold=0.02,
        #                                    allowable_rare_percentage=0.1)

        return self



    def transform(self):
        """Transforms new data per the treatment design plans."""
        self.treated_df_ = self.df_.copy()
        # fill in missing values
        self.df_ = self._fill_missing_values()
        pass


    def _get_dtypes(self):
        """Finds the dtypes in the dataset."""
        features = self.df_.columns
        if self.target is not None:
            features = features.drop(labels=[self.target])
        if self.index_feature is not None:
            features = features.drop(labels=[self.index_feature])
        objs = self.df_.loc[:, features]\
            .select_dtypes(include=OBJECT_DTYPES).columns
        cats = self.df_.loc[:, features]\
            .select_dtypes(include=CATEGORICAL_DTYPES).columns
        bool = self.df_.loc[:, features]\
            .select_dtypes(include=BOOL_DTYPES).columns
        ints = self.df_.loc[:, features]\
            .select_dtypes(include=INT_DTYPES).columns
        floats = self.df_.loc[:, features]\
            .select_dtypes(include=FLOAT_DTYPES).columns
        dts = self.df_.loc[:, features]\
            .select_dtypes(include=DATETIME_DTYPES).columns
        dttz = self.df_.loc[:, features]\
            .select_dtypes(include=DATETIMETZ_DTYPES).columns
        tds = self.df_.loc[:, features]\
            .select_dtypes(include=TIMEDELTAS).columns
        return objs, cats, bool, ints, floats, dts, dttz, tds


    def _get_removable_features(self):
        """Adds the currently unsupported columns to the excluded
        features attribute."""
        removed_features = list()
        removed_features += self.datetime_timezone_features_.tolist()
        removed_features += self.datetime_features_.tolist()
        removed_features += self.timedelta_features_.tolist()
        return removed_features


    def _find_hidden_numerics(self):
        """Finds hidden numerical dtypes among the object dtypes."""
        hidden_ints = \
            self.df_.loc[:, self.object_features_] \
                .fillna(int(self.numerical_fill_value)) \
                .apply(lambda x: pd.to_numeric(x, errors="ignore"))\
                .select_dtypes(include=INT_DTYPES) \
                .columns\
                .tolist()
        hidden_floats = \
            self.df_.loc[:, self.object_features_] \
                .fillna(self.numerical_fill_value) \
                .apply(lambda x: pd.to_numeric(x, errors="ignore")) \
                .select_dtypes(include=FLOAT_DTYPES) \
                .columns \
                .tolist()
        return hidden_ints, hidden_floats


    def _find_zero_variance_features(self):
        """Detects zero-variance features which contain no useful
        information for downstream algorithms."""
        zero_var = list()
        # check numerical features for zero-variance
        for feature in self.integer_features_.tolist() + \
                       self.float_features_.tolist():
            uniques = self.df_.loc[:, feature].unique()
            if len(uniques) < 2:
                zero_var.append(feature)
                continue
        # check object features for zero-variance
        for feature in self.object_features_.tolist():
            vals = self.df_.loc[:, feature].values
            try:
                vals = [v.upper().strip() for v in vals]
                uniques = pd.unique(vals)
                if len(uniques) < 2:
                    zero_var.append(feature)
                    continue
            except:
                continue
        return zero_var


    def _find_nan_features(self):
        """Identifies features with missing values."""
        nan_features = list()
        for feature in self.df_.columns.tolist():
            if self.df_.loc[:, feature].isnull().any().sum() != 0:
                nan_features.append(feature)
        return nan_features


    def _get_treatment_features(self):
        """Creates a list of features that will be used for treatment
        design."""
        treatment_features = list()
        # keep all features not excluded up to this point
        treatment_features += \
            self.df_.loc[:, ~self.df_.columns.isin(
                self.excluded_features_ +
                self.removable_features_ +
                self.zero_variance_features_)]\
                .columns\
                .tolist()
        return treatment_features


    def _cast_to_int(self):
        """Attempts to cast float features to int which will then be
        treated as categorical further downstream."""
        # we just want to know which ones can safely be cast to ints
        if self.is_fitted_ is False:
            castables = list()
            for feature in self.float_features_.tolist() + \
                           self.hidden_floats_:
                try:
                    int_count = self.df_.loc[:, feature]\
                        .fillna(value=self.numerical_fill_value)\
                        .apply(lambda x: x.is_integer()).sum()
                    if int_count == self.df_.shape[0]:
                        castables.append(feature)
                except:
                    pass
            return castables
        # we are in the transform method and actually want to apply
        # the transformation
        else:
            self.treated_df_.loc[:, self.integer_castables_] = \
                self.treated_df_ \
                    .loc[:, self.integer_castables_]\
                    .astype("int")
            return self


    # def _cast_to_category(self):
    #     """Attempts to cast int features to categories."""
    #     # if we're at the fit stage, we just want to know which ones
    #     # can safely be cast to categories
    #     if self.is_fitted_ is False:
    #         castables = list()
    #         for feature in self.integer_features_.tolist() + \
    #                        self.hidden_ints_:
    #             try:
    #                 cat_count = self.df_.loc[:, feature] \
    #                     .apply(lambda x: x.is_integer()).sum()
    #                 castables.append(feature)
    #             except:
    #                 pass
    #         return castables
    #     # go ahead and actually cast it
    #     else:
    #         self.df_.loc[:, self.category_castables_] = \
    #             self.df_ \
    #                 .loc[:, self.category_castables_]\
    #                 .astype("category")
    #         return self

    def find_high_cardinality_features(dataframe=None,
                                       object_features=None,
                                       rare_level_threshold=0.02,
                                       allowable_rare_percentage=0.1):
        """Returns a list of high-cardinality features and those which
        can be cast to categorical."""
        # container to hold high-cardinality features
        high_cardinality = list()

        # container to hold categorical features; we'll actually convert
        # these to the "category" dtype to save memory
        categorical = list()

        # do some simple checks first
        for feature in object_features:
            # counter that we'll use to determine if there are too many
            # unique levels for the current feature
            rare_level_row_counts = dict()

            # unique features for the
            unique_vals = dataframe.loc[:, feature].unique()

            # if every row is a unique value, add it to the list and skip
            # to the next feature
            if len(unique_vals) == dataframe.shape[0]:
                high_cardinality.append(feature)
                continue

            # get value counts for each level
            gb = dataframe.groupby(feature).agg({feature: "count"})
            gb.columns = ["count"]
            gb = gb.reset_index()
            gb.columns = ["level", "count"]
            keys = gb.loc[:, "level"].values
            vals = gb.loc[:, "count"].values

            # if the level occurs below the threshold, add its number of
            # rows to the dict
            for key, val in zip(keys, vals):
                if val / dataframe.shape[0] <= rare_level_threshold:
                    rare_level_row_counts[key] = val

            # sum the values of the rare dict and calculate what
            # percentage of the dataset is occupied by rare levels for
            # the current feature
            rare_count = sum(rare_level_row_counts.values())
            rare_percent = rare_count / dataframe.shape[0]

            # if the rare percent is greater than the allowable percent,
            # this is a high-cardinality feature
            if rare_percent > allowable_rare_percentage:
                high_cardinality.append(feature)
            # or it is eligible to be encoded as a "category" dtype
            elif rare_percent <= allowable_rare_percentage:
                categorical.append(feature)

        return high_cardinality, categorical

    def _reindex_target(self):
        """Moves the target feature to the end of the dataframe."""
        # move `target` column to the end (if present)
        features = self.df_.columns.tolist()
        insert_loc = len(features) - 1
        features.insert(insert_loc,
                        features.pop(features.index(self.target)))
        dataframe = self.df_.reindex(columns=features)
        return dataframe


    def _fill_missing_values(self):
        """Replaces missing values by dtype."""
        # replace missing values
        df = self.df_.copy()
        df.loc[:, self.object_features_] = \
            df.loc[:, self.object_features_] \
                .fillna(value=self.categorical_fill_value)
        df.loc[:, self.integer_features_] = \
            df.loc[:, self.integer_features_] \
                .fillna(value=int(self.numerical_fill_value))
        df.loc[:, self.float_features_] = \
            df.loc[:, self.float_features_] \
                .fillna(value=self.numerical_fill_value)
        return df





    # def _cast_to_float(self):




    # def _cast_to_categories(self):