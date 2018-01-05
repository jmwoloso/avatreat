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
