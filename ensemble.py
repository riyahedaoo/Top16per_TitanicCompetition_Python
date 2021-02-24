from sklearn.ensemble import VotingClassifier
evc = VotingClassifier(estimators=[('gbm',modelgbm),('rf',modelrf),
                                   ('svm',modelsv),('dt',modeldt)], 
                       voting='hard')
evc.fit(x_train,y_train)
print(evc.score(x_test,y_test))

