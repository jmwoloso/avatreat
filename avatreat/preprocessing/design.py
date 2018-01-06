from ..utils.routines.fit import find_hidden_dtypes, \
    fill_missing_values, find_zero_variance_features, \
    get_treatment_features, reindex_target, cast_to_int, \
    get_column_dtypes, find_high_cardinality_features


class TreatmentDesign(object):
    def __init__(self, index_features=None,
                 datetime_features=None,
                 target=None, target_type="categorical",
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
        index_features:   one of {list of features, None};
        default=None; list of features that are purely for
        identification/indexing and will be excluded from treatment
        design.

        datetime_features:   one of {list of features, None};
        default=None; list of datetime-like features; will be
        excluded from treatment design.

        NOTE: treatment design will fail if the dataset contains
        datetime features but none are specified.

        target:   one of {str, None}; default=None; name of the
        target feature in the dataset.

        target_type:   one of {"categorical", "numerical"};
        default="categorical"; whether the design is for a regression or
        classification target.

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

        exclude_zero_variance_features: bool; default=True; whether
        to eliminate features with no variance from treatment design
        as they contain no useful information.


        Attributes
        ----------
        df_:   copy of the supplied dataframe used for modification

        blacklist_: features to be excluded from treatment design due
        to having zero-variance.

        treatment_features_:   features to be used for treatment design.

        int_features_:   integer features of the dataframe.

        float_features_:   float features of the dataframe.

        object_features_:   object/str features of the dataframe.

        bool_features_: boolean features of the dataframe.

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
        self.index_features = list(index_features) \
            if index_features is not None \
            else list()
        # if no datetime features are supplied, replace None with
        # empty list
        self.datetime_features = list(datetime_features) \
            if datetime_features is not None \
            else list()
        self.target = target
        self.target_type = target_type
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
        self.exclude_zero_variance_features = exclude_zero_variance_features


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

        # find any hidden dtypes within the object dtypes and
        # downcast numerical features if specified
        self.df_ = find_hidden_dtypes(dataframe=self.df_)

        # fill in missing values
        self.df_ = fill_missing_values(dataframe=self.df_,
                                       numerical_fill_value=self.numerical_fill_value,
                                       categorical_fill_value=self.categorical_fill_value)

        self.blacklist_ = \
            find_zero_variance_features(dataframe=self.df_,
            exclude_zero_variance_features=self.exclude_zero_variance_features,
                                        categorical_fill_value=self.categorical_fill_value)

        # get a list of the features to be included in treatment
        # design; also removes zero-variance features
        self.treatment_features_ = get_treatment_features(
            dataframe=self.df_,
            id_features=self.index_features,
            datetime_features=self.datetime_features,
            target=self.target,
            blacklist=self.blacklist_)

        # move the target feature to the end of the dataframe
        self.df_ = reindex_target(dataframe=self.df_,
                                  target=self.target)

        # try and cast float features to ints
        self.df_.loc[:, self.treatment_features_] = \
            cast_to_int(dataframe=self.df_,
                        treatment_features=self.treatment_features_)

        # inspect object features for booleans
        # objs = self.df_.select_dtypes(include=["object"]).columns


        # get the features by dtype
        self.int_features_, \
        self.float_features_, \
        self.object_features_, \
        self.bool_features_ = \
            get_column_dtypes(dataframe=self.df_,
                              treatment_features=self.treatment_features_)

        # find high-cardinality features from within object features
        self.high_cardinality_features_, self.categorical_features_ = \
            find_high_cardinality_features(dataframe=self.df_,
                                           object_features=self.object_features_,
                                           rare_level_threshold=0.02,
                                           allowable_rare_percentage=0.1)





    def transform(self):
        """Transforms new data per the treatment design plans."""
        pass