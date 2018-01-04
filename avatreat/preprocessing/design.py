


class TreatmentDesign(object):
    """Class for designing data treatments."""
    def __init__(self, id_features=None, datetime_features=None,
                 target=None, target_type="categorical",
                 missing_numerical_strategy="systematically",
                 numerical_fill_value=-1.0, ints_to_categorical=True,
                 categorical_fill_value="NA", rare_level_threshold=0.02,
                 allowable_rare_percentage=0.1,
                 variable_significance_threshold="1/n",
                 smoothing_factor=0.0):
        """

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

        Attributes
        ----------

        """
        pass