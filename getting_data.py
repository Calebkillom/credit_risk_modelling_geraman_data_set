import pandas as pd

""" Read the CSV file into a DataFrame """
df = pd.read_csv("german_credit.csv")

""" Display the first few rows of the DataFrame to inspect the data """
print("First few rows of the DataFrame:")
print(df.head())

""" Display the shape of the DataFrame """
print("\nShape of the DataFrame:", df.shape)
