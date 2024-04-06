from ucimlrepo import fetch_ucirepo
import pandas as pd

""" Fetch Dataset """
statlog_german_credit = fetch_ucirepo(id=44)

""" data as pandas dataframe """
x = statlog_german_credit.data.features
y = statlog_german_credit.data.targets

""" metadata """
print(statlog_german_credit.metadata)

print("The switch before")

""" variable information """
print(statlog_german_credit.variables)

# Extract feature names from the DataFrame
feature_names = statlog_german_credit.variables.loc[1:, 'name'].tolist()

# Print feature names
print("Feature Names:", feature_names)