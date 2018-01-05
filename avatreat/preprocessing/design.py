from ..utils.routines import exclude_features, reindex_target, \
    get_column_dtypes, get_high_cardinality_features


class TreatmentDesign(object):
    def __init__(self, id_features=None,
                 datetime_features=None,
                 target=None, target_type="categorical",
                 missing_numerical_strategy="systematically",
                 numerical_fill_value=-1.0, ints_to_categorical=True,
                 categorical_fill_value="NA", rare_level_threshold=0.02,
                 allowable_rare_percentage=0.1,
                 variable_significance_threshold="1/n",
                 smoothing_factor=0.0,
                 dtype_compression=False):
        """
        Class for designing data treatments.

        Parameters
        ----------
        id_features :   one of {list of features, None}; default=None;
        list of features that are purely for identification and will be
        excluded from treatment design.

        datetime_features   :   one of {list of features, None};
        default=None; list of datetime-like features; will be
        excluded from treatment design.

        target  :   one of {str, None}; default=None; name of the
        target feature in the dataset.

        target_type :   one of {"categorical", "numerical"};
        default="categorical"; whether the design is for a regression or
        classification target.

        missing_numerical_strategy  :   one of {"systematically",
        "random"}; default="systematically"; the strategy to employ
        when filling in missing numerical values; used in conjunction
        with `numerical_fill_value`; if strategy is set to "random" then
        `numerical_fill_value` will be set to "mean" automatically
        regardless of the value provided.

        NOTE: the default strategy of "systematically" is the safe
        choice as importance is often implicit in missing values.

        numerical_fill_value    :   one of {float, "mean"};
        default=-1.0; value to fill in missing numerical rows with;
        used in conjunction with `missing_numerical_strategy`.

        ints_to_categorical :   bool; default=True; whether to
        attempt converting numeric features with dtype=int to
        categorical features; if True, rules applying to categorical
        features (`categorical_fill_value`, `rare_level_threshold`,
        `allowable_rare_percentage`) will be similarly applied.

        categorical_fill_value  :   str; default="NA"; string to be
        used to fill in missing categorical levels.

        rare_level_threshold    :   float; default=0.02; the
        percetage of the dataset that a categorical level must not
        exceed in order to be classified as (and treated accordingly) a
        rare level.

        NOTE: rare levels will be pooled together when creating
        indicators for categorical features.

        allowable_rare_percentage   :   float; default=0.1; total
        percentage of the dataset that all "rare" levels of a
        categorical feature must not exceed in order to pool rare
        levels; aims to prevent categorical features with
        high-cardinality (many levels) from having rare levels pooled
        together since most levels are rare in that case; features
        exceeding this level will be classified as having
        high-cardinality and will not have indicators created.

        variable_significance_threshold :   one of {"1/n", float,
        None}; default="1/n" features; significance level that a
        feature must achieve in order to remain in the final treated
        dataset; if `None` all features will be kept.

        smoothing_factor    :   float; default=0.0; number of
        pseudo-observations to add as a Laplace smoothing factor;
        reduces the range of predictions of rare levels.

        dtype_compression   :   bool; default=True; whether to
        downcast column dtypes (e.g. np.int64 -> np.int8);
        downcasting depends on the range of values present for a
        given column and may not always be possible/successful;
        useful for very large datasets as it helps reduce the
        in-memory footprint of the data.


        Attributes
        ----------
        self.df_    :   pandas.DataFrame instance with ID and
        datetime features removed.

        """
        # if no id features are supplied, replace None with empty list
        self.id_features = list(id_features) \
            if id_features is not None \
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


    def fit(self, DF):
        """
        Runs the treatment design processes on the dataset.

        Parameters
        ----------
        DF   :   pandas.DataFrame instance; dataframe used to to
        design treatments.


        Returns
        -------
        self    :   object
        """
        # remove the id and datetime features
        self.df_ = exclude_features(dataframe=DF,
                                    id_features=self.id_features,
                                    datetime_features=self.datetime_features)

        # move the target feature to the end of the dataframe
        self.df_ = reindex_target(dataframe=self.df_,
                                  target=self.target)

        # get the features by dtype
        self.int_features_, self.float_features_, self.object_features_ = \
            get_column_dtypes(dataframe=self.df_)

        # find high-cardinality features from within object features
        self.high_cardinality_features_, self.categorical_features_ = \
            get_high_cardinality_features(dataframe=self.df_,
                                          object_features=self.object_features_,
                                          rare_level_threshold=0.02,
                                          allowable_rare_percentage=0.1)





    def transform(self):
        """Transforms new data per the treatment design plans."""
        pass