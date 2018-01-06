import pandas as pd



def exclude_features(dataframe=None, id_features=None,
                     datetime_features=None):
    """Removes ID and Datetime features and returns the modified
    dataframe."""
    # keep all features except id, datetime
    return dataframe.loc[:, ~dataframe.columns.isin(id_features +
                                                    datetime_features)]


def reindex_target(dataframe=None, target=None):
    """Moves the target feature to the end of the dataframe."""
    # move `target` column to the end (if present)
    if target is not None:
        target_vals = dataframe.loc[:, target].values
        dataframe = dataframe.drop(labels=[target],
                                   axis=1)

        dataframe.loc[:, target] = target_vals
    return dataframe


def cast_float_to_int(dataframe=None,
                      fill_value=-1):
    """Attempts to cast float features to int which will then be
    treated as categorical further downstream."""
    int_dtypes = ["int_", "intc", "intp", "int8", "int16", "int32",
                  "int64", "uint8", "uint16", "uint32", "uint64"]

    float_dtypes = ["float_", "float16", "float32", "float64"]

    floats = dataframe.select_dtypes(include=["float"]).columns

    # fill missing values with `fill_value` and then attempt to
    # convert to int
    dataframe.loc[:, floats] = dataframe.loc[:, floats]\
        .apply(lambda x: pd.to_numeric(x, downcast="integer"))

    return dataframe


def get_column_dtypes(dataframe=None):
    """Returns lists of the various dtypes that will be used further
    downstream in the processing pipeline."""
    int_dtypes = ["int_", "intc", "intp", "int8", "int16", "int32",
                  "int64", "uint8", "uint16", "uint32", "uint64"]

    float_dtypes = ["float_", "float16", "float32", "float64"]
    int_features = dataframe.select_dtypes(include=int_dtypes)
    float_features = dataframe.select_dtypes(include=float_dtypes)
    object_features = dataframe.select_dtypes(include=["object"])
    bool_features = dataframe.select_dtypes(include=["bool_"])
    return int_features, float_features, object_features, bool_features


def get_high_cardinality_features(dataframe=None,
                                  object_features=None,
                                  rare_level_threshold=0.02,
                                  allowable_rare_percentage=0.1):
    """Returns a list of high-cardinality features."""
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


