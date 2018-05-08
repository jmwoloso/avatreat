import collections

class Treatments(collections.OrderedDict):
    def __init__(self, treatment_plan=None):
        """Class that serves as a lookup table mapping strings to
        functions that are used to transform a dataframe."""
        super().__init__()



