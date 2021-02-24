import pandas as pd

df = pd.read_csv("C:/Users/Saniya and Family/Downloads/titanic_train.csv")
print(df.isna().sum())

data= df.fillna(df.mean())
data['Embarked']=data['Embarked'].fillna("S")

print(data.isna().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Sex']=le.fit_transform(data['Sex'])


data=data.drop(columns=['Cabin','Name','PassengerId','Ticket','Parch','SibSp','Embarked'])
x=data.drop(columns=['Survived'])
y=data['Survived']

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y)
