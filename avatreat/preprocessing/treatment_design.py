import numpy as np
import pandas as pd

from avatreat.utils.constants import OBJECT_DTYPES, INT_DTYPES, \
    FLOAT_DTYPES, NUMERICAL_DTYPES, DATETIME_DTYPES, TIMEDELTAS, \
    CATEGORICAL_DTYPES, DATETIMETZ_DTYPES, BOOL_DTYPES

from avatreat.utils.treatment_design import \
    reindex_target, cast_to_int, \
    find_high_cardinality_features


class TreatmentDesign(object):
    def __init__(self, index_feature=None,
                 target=None, target_type="categorical",
                 find_hidden_dtypes=False,
                 missing_numerical_strategy="systematically",
                 numerical_fill_value=-1.0, ints_to_categorical=True,
                 categorical_fill_value="NA", rare_level_threshold=0.02,
                 allowable_rare_percentage=0.1,
                 variable_significance_threshold="1/n",
                 smoothing_factor=0.0,
                 dtype_compression=False,
                 exclude_zero_variance_features=True):
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

        find_hidden_dtypes: bool; default=False; whether to try and
        find numeric features among any features that identify as the
        object dtype.

        exclude_zero_variance_features: bool; default=True; whether
        to eliminate features with no variance from treatment design
        as they contain no useful information.

        missing_numerical_strategy:   one of {"systematically",
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

        ints_to_categorical:   bool; default=True; whether to
        attempt converting numeric features with dtype=int to
        categorical features; if True, rules applying to categorical
        features (`categorical_fill_value`, `rare_level_threshold`,
        `allowable_rare_percentage`) will be similarly applied.

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
        design

        treatment_features_:   features to be used for treatment design.

        object_features_:   features with the object dtype.

        categorical_features_:  features with the category dtype.

        boolean_features_:  features with the bool_ dtype.

        integer_features_:  features with the integer dtype.

        float_features_:    features with the float dtype.

        datetime_features_: features with the datetime dtype.

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
        self.find_hidden_dtypes = find_hidden_dtypes
        self.exclude_zero_variance_features = exclude_zero_variance_features
        self.missing_numerical_strategy = missing_numerical_strategy
        self.numerical_fill_value = numerical_fill_value if \
            missing_numerical_strategy == "systematically" else "mean"
        self.ints_to_categorical = ints_to_categorical
        self.categorical_fill_value = categorical_fill_value
        self.rare_level_threshold = rare_level_threshold
        self.allowable_rare_percentage = allowable_rare_percentage
        self.variable_significance_threshold = variable_significance_threshold
        self.smoothing_factor = smoothing_factor
        self.dtype_compression = dtype_compression



    def fit(self, DF):
        """
        Runs the treatment design processes on the dataset.

        Parameters
        ----------
        DF   :   pandas.DataFrame instance; dataframe used to to
        design treatments.

        """
        # copy the dataframe for modification
        self.df_ = DF.copy()

        # list of columns to exclude from treatment design
        self.excluded_features_ = list()

        # exclude target and index if passed in
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

        # TODO: add support for feature engineering with datetimes
        self.excluded_features_ += \
            self.datetime_timezone_features_.tolist()
        self.excluded_features_ += self.datetime_features_.tolist()
        self.excluded_features_ += self.timedelta_features_.tolist()

        # find any hidden dtypes within the object dtypes and
        # downcast numerical features if specified
        # if self.find_hidden_dtypes is True:
        #     #TODO: on hold for now
        #     self._find_hidden_dtypes()

        # fill in missing values
        self._fill_missing_values()

        # find features with zero variance and drop them, but note
        # which ones we dropped
        if self.exclude_zero_variance_features is True:
            self.zero_variance_features_ = \
                self._find_zero_variance_features()
            if len(self.zero_variance_features_) > 0:
                self.excluded_features_\
                    .append(*self.zero_variance_features_)

        # get a list of the features to be included in treatment
        # design; also removes zero-variance features
        self.treatment_features_ = \
                self._get_treatment_features()

        # move the target feature to the end of the dataframe
        # self.df_ = self._reindex_target(dataframe=self.df_,
        #                           target=self.target)

        # # try and cast float features to ints
        # self.df_.loc[:, self.treatment_features_] = \
        #     cast_to_int(dataframe=self.df_,
        #                 treatment_features=self.treatment_features_)
        #
        # # inspect object features for booleans
        # # objs = self.df_.select_dtypes(include=["object"]).columns
        #
        #
        # # get the features by dtype
        # self.int_features_, \
        # self.float_features_, \
        # self.object_features_, \
        # self.bool_features_ = \
        #     get_column_dtypes(dataframe=self.df_,
        #                       treatment_features=self.treatment_features_)
        #
        # # find high-cardinality features from within object features
        # self.high_cardinality_features_, self.categorical_features_ = \
        #     find_high_cardinality_features(dataframe=self.df_,
        #                                    object_features=self.object_features_,
        #                                    rare_level_threshold=0.02,
        #                                    allowable_rare_percentage=0.1)

        return self



    def transform(self):
        """Transforms new data per the treatment design plans."""
        pass


    def _get_dtypes(self):
        """Finds the dtypes and creates a mapping."""
        objs = self.df_.select_dtypes(include=OBJECT_DTYPES).columns
        ints = self.df_.select_dtypes(include=INT_DTYPES).columns
        floats = self.df_.select_dtypes(include=FLOAT_DTYPES).columns
        dts = self.df_.select_dtypes(include=DATETIME_DTYPES).columns
        tds = self.df_.select_dtypes(include=TIMEDELTAS).columns
        cats = self.df_\
            .select_dtypes(include=CATEGORICAL_DTYPES).columns
        dttz = self.df_\
            .select_dtypes(include=DATETIMETZ_DTYPES).columns
        bool = self.df_.select_dtypes(include=BOOL_DTYPES).columns
        return objs, cats, bool, ints, floats, dts, dttz, tds

    def _find_hidden_dtypes(self):
        """Finds hidden dtypes among the object dtypes."""
        #TODO: on hold for now
        # try and convert the object dtypes to numeric
        new_ints = list()
        new_floats = list()
        for feature in self.object_features_:
            try:
                # attempt to cast object dtypes to ints
                self.df_.loc[:, feature] = \
                    self.df_.loc[:, feature] \
                        .applymap(lambda x: np.int(x))
                new_ints.append(feature)
            except ValueError as e:
                try:
                    # int casting failed so try floats now
                    self.df_.loc[:, feature] = \
                        self.df_.loc[:, feature] \
                            .applymap(lambda x: np.float(x))
                    new_floats.append(feature)
                except ValueError as e:
                    pass

        if len(new_ints) > 0:
            # update our attributes
            self.object_features_ = \
                self.object_features_.drop(labels=new_ints)
            self.integer_features_ = \
                pd.Index(self.integer_features_.tolist() + new_ints)

        if len(new_floats) > 0:
            # update our attributes
            self.object_features_ = \
                self.object_features_.drop(labels=new_floats)
            self.float_features_ = \
                pd.Index(self.float_features_.tolist + new_floats)

        # iterate through the object columns again and check for
        # booleans disguised as strings
        new_bools = list()
        for feature in self.object_features_:
            uniques = self.df_.loc[:, feature].unique().tolist()
            if len(uniques) != 2:
                continue
            else:
                try:
                    uniques = [u.capitalize().strip() for u in uniques]
                    bools = ["True", "False"]
                    if set(bools) == set(uniques):
                        # first capitalize the values
                        self.df_.loc[:, feature] = \
                            self.df_.loc[:, feature] \
                                .map(lambda x: x.capitalize().strip())
                        # then map to True/False
                        self.df_.loc[:, feature] = \
                            self.df_.loc[:, feature].map({"True": True,
                                                          "False": False})
                        new_bools.append(feature)
                except ValueError as e:
                    continue
        if len(new_bools) > 0:
            # update the attributes
            self.object_features_ = \
                self.object_features_.drop(labels=new_bools)

            self.boolean_features_ = \
                pd.Index(self.boolean_features_.tolist() + new_bools)

        return self


    def _fill_missing_values(self):
        """Replaces missing values by dtype."""
        # replace missing values
        self.df_.loc[:, self.object_features_] = \
            self.df_.loc[:, self.object_features_]\
                .fillna(value=self.categorical_fill_value)
        self.df_.loc[:, self.integer_features_] = \
            self.df_.loc[:, self.integer_features_]\
                .fillna(value=int(self.numerical_fill_value))
        self.df_.loc[:, self.float_features_] = \
            self.df_.loc[:, self.float_features_]\
                .fillna(value=self.numerical_fill_value)
        return self


    def _find_zero_variance_features(self):
        """Detects zero-variance features which contain no useful
            information for downstream algorithms."""
        blacklist = list()
        # check numeric features for zero-variance
        for feature in self.integer_features_.tolist() + \
                       self.float_features_.tolist():
            uniques = self.df_.loc[:, feature].unique()
            if len(uniques) < 2:
                blacklist.append(feature)
                continue
        # check object features for zero-variance
        for feature in self.object_features_:
            vals = self.df_.loc[:, feature].values
            vals = [v.upper().strip() for v in vals]
            uniques = pd.unique(vals)
            if len(uniques) < 2:
                blacklist.append(feature)
                continue
        return blacklist


    def _get_treatment_features(self):
        """Creates a list of features that will be used for treatment
        design."""
        treatment_features = list()
        # keep all features not excluded up to this point
        treatment_features += \
            self.df_.loc[:, ~self.df_.columns.isin(
                self.excluded_features_)].columns.tolist()
        return treatment_features


    # def _reindex_target(self):
