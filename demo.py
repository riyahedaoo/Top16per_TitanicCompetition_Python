import pandas as pd

data = pd.read_csv("C:/Users/Saniya and Family/Downloads/titanic_train.csv")
print(data.head())
print(data.sort_values(by="Survived"))
print(data.describe())
print(data.Survived.value_counts())
print(data[~data.Age.isna()]) #removes age NA
print(data.isna().any())
print(data.dropna())
print(data.Embarked.fillna("S"))
print(data.drop_duplicates())

print(data.isna().sum())




