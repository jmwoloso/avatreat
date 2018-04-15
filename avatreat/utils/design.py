import pandas as pd

from avatreat.utils.constants import INT_DTYPES, FLOAT_DTYPES, NUMERICAL_DTYPES

def get_dtypes(dataframe=None):
    """Finds the dtypes and creates a mapping."""
    objs = dataframe.select_dtypes(include=["object"]).columns
    ints = dataframe.select_dtypes(include=[INT_DTYPES]).columns
    floats = dataframe.select_dtypes(include=[FLOAT_DTYPES]).columns
    dts = dataframe.select_dtypes(include=[])
    return objs, ints, floats, dts


def find_hidden_dtypes(dataframe=None):
    """Finds hidden dtypes among the object dtypes."""
    objs = dataframe.select_dtypes(include=["object"]).columns.tolist()

    for feature in objs:
        try:
            # attempt to cast object dtypes to numeric
            dataframe.loc[:, feature] = \
                dataframe.loc[:, feature]\
                    .apply(lambda x: pd.to_numeric(x))
        except ValueError as e:
            pass

    # iterate through the remaining columns and check for booleans
    # disguised as strings
    objs_ = dataframe.select_dtypes(include=["object"]).columns.tolist()
    for feature in objs_:
        uniques = dataframe.loc[:, feature].unique().tolist()
        if len(uniques) != 2:
            continue
        else:
            try:
                uniques = [u.capitalize().strip() for u in uniques]
                bools = ["True", "False"]
                if set(bools) == set(uniques):
                    # first capitalize the values
                    dataframe.loc[:, feature] = \
                        dataframe.loc[:, feature]\
                            .map(lambda x: x.capitalize().strip())
                    dataframe.loc[:, feature] = \
                        dataframe.loc[:, feature].map({"True":  True,
                                                       "False": False})
            except ValueError as e:
                continue


    #TODO [ENH]: add discovery for remaining dtypes (datetime, timedelta, bool)

    return dataframe


def fill_missing_values(dataframe=None,
                        numerical_fill_value=None,
                        categorical_fill_value=None,
                        missing_numerical_strategy=None):
    """Replaces missing values by dtype."""
    objs = dataframe.select_dtypes(include=["object"]).columns
    ints = dataframe.select_dtypes(include=INT_DTYPES).columns
    floats = dataframe.select_dtypes(include=FLOAT_DTYPES).columns

    # replace missing values
    dataframe.loc[:, objs] = dataframe.loc[:, objs]\
        .fillna(value=categorical_fill_value)
    if
    dataframe.loc[:, ints] = dataframe.loc[:, ints]\
        .fillna(value=int(numerical_fill_value))
    dataframe.loc[:, floats] = dataframe.loc[:, floats]\
        .fillna(value=numerical_fill_value)

    return dataframe


def find_zero_variance_features(dataframe=None,
                                exclude_zero_variance_features=True,
                                categorical_fill_value="NA"):
    """Detects zero-variance features which contain no useful
    information for downstream algorithms."""
    blacklist = list()

    if exclude_zero_variance_features is False:
        return blacklist

    elif exclude_zero_variance_features is True:
        # numeric features with missing values show up as object dtypes
        # attempt to cast anything to a number that we can
        for feature in dataframe.columns:
            try:
                dataframe.loc[:, feature] = \
                    dataframe.loc[:, feature]\
                        .apply(lambda x: pd.to_numeric(x))

            # thrown when encountering strings
            except ValueError as e:
                continue

            # thrown when encountering datetimes
            except TypeError as e:
                continue

        # check numeric features for zero-variance
        for feature in dataframe\
                .select_dtypes(include=NUMERICAL_DTYPES).columns:
            uniques = dataframe.loc[:, feature].unique()
            if len(uniques) < 2:
                blacklist.append(feature)
                continue

        # check object features for zero-variance
        for feature in dataframe.select_dtypes(include=["object"]).columns:
            vals = dataframe.loc[:, feature].values
            vals = [v.upper().strip() for v in vals]
            uniques = pd.unique(vals)
            if len(uniques) < 2:
                blacklist.append(feature)
                continue
        return blacklist


def get_treatment_features(dataframe=None, id_features=None,
                           datetime_features=None, target=None,
                           blacklist=None):
    """Creates a list of features that will be used for treatment
    design."""
    # keep all features except id, datetime, target and blacklisted
    treatment_features = \
        dataframe.loc[:, ~dataframe.columns.isin(id_features +
                                                 datetime_features +
                                                 [target] +
                                                 blacklist)].columns.tolist()
    return treatment_features


def reindex_target(dataframe=None, target=None):
    """Moves the target feature to the end of the dataframe."""
    # move `target` column to the end (if present)
    if target is not None:
        features = dataframe.columns.tolist()
        insert_loc = len(features) - 1
        features.insert(insert_loc,
                        features.pop(features.index(target)))
        dataframe = dataframe.reindex(columns=features)

    return dataframe


def cast_to_int(dataframe=None, treatment_features=None):
    """Attempts to cast float features to int which will then be
    treated as categorical further downstream."""
    floats = \
        dataframe.loc[:, treatment_features]\
            .select_dtypes(include=FLOAT_DTYPES).columns

    dataframe.loc[:, floats] = dataframe.loc[:, floats]\
        .apply(lambda x: pd.to_numeric(x, downcast="integer"))

    return dataframe


def get_column_dtypes(dataframe=None, treatment_features=None):
    """Returns lists of the various dtypes that will be used further
    downstream in the processing pipeline."""
    int_features = dataframe.loc[:, treatment_features]\
        .select_dtypes(include=INT_DTYPES)\
        .columns\
        .tolist()

    float_features = dataframe.loc[:, treatment_features]\
        .select_dtypes(include=FLOAT_DTYPES)\
        .columns\
        .tolist()

    object_features = dataframe.loc[:,
                      treatment_features]\
        .select_dtypes(include=["object"])\
        .columns\
        .tolist()

    bool_features = dataframe.loc[:,
                    treatment_features]\
        .select_dtypes(include=["bool_"])\
        .columns\
        .tolist()

    return int_features, float_features, object_features, bool_features


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


