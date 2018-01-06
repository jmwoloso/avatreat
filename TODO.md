# `[feature enhancements]`
1. add I/O
   1. read files remotely
   2. ability to specify dask dataframe use for everything (useful for if you know that your data will not fit in memory)
      1. expose `dd.read_csv` keywords, etc.
2. add serialization functionality for `TreatmentDesign`
3. add functionality to handle datetime columns and include them in treatment design??
4. allow passing in dict of {column_name: dtype}
5. add/allow passing in suffix dict to use for column renaming (e.g. {"integer": "n", "float": "f"})
6. add input validation
7. add functionality to discover other hidden dtypes besides numeric
8. add CI to repo

# `[tests]`
1. add tests for `preprocessing.categorical`
2. add tests for `preprocessing.numerical`
