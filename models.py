#Logistic Regression
from sklearn.linear_model import LogisticRegression
modellm = LogisticRegression(solver='liblinear', random_state=0)
modellm.fit(x_train, y_train)
y_pred=modellm.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("LM" ,end="  ")
print(modellm.score(x_test,y_test))


#Decision Tree
from sklearn import tree
modeldt = tree.DecisionTreeClassifier(max_depth=3)
modeldt.fit(x_train, y_train)
y_pred=modeldt.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("DT" ,end="  ")
print(modeldt.score(x_test,y_test))


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()
modelnb.fit(x_train, y_train)
y_pred=modelnb.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("NB" ,end="  ")
print(modelnb.score(x_test,y_test))


#SVM
from sklearn.svm import SVC
modelsv=SVC( C=10000)
modelsv.fit(x_train, y_train)
y_pred=modelsv.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("SVM" ,end="  ")
print(modelsv.score(x_test,y_test))



#RF
from sklearn.ensemble import RandomForestClassifier
modelrf = RandomForestClassifier(n_estimators=200,max_depth=3)
modelrf.fit(x_train, y_train)
y_pred=modelrf.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("RF" ,end="  ")
print(modelrf.score(x_test,y_test))


#GBM
from sklearn.ensemble import GradientBoostingClassifier
modelgbm=GradientBoostingClassifier()
modelgbm.fit(x_train,y_train)
y_pred=modelgbm.predict(x_test)
#print(confusion_matrix(y_test, y_pred)) 
print("GBM" ,end="  ")
print(modelgbm.score(x_test,y_test))