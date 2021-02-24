import pandas as pd
testd = pd.read_csv("C:/Users/Saniya and Family/Downloads/titanic_test.csv")
submit = pd.read_csv("C:/Users/Saniya and Family/Downloads/titanic_submit.csv")
testd= testd.fillna(testd.mean())
testd=testd.drop(columns=['Cabin','Name','PassengerId','Ticket'])
print(testd.isna().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
testd['Sex']=le.fit_transform(testd['Sex'])
testd['Embarked']=le.fit_transform(testd['Embarked'])


p = evc.predict(testd)
submit.Survived=p
submit.to_csv("titanic.csv")