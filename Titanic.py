import pandas as pd
import numpy as np
from pathlib import Path  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


dataSet  = pd.read_csv("~/Documents/Kaggle Projects/Titanic Project/train.csv")
# print (dataSet.columns)

trainSet = dataSet.drop(columns = ['Name','PassengerId', 'Ticket', 'Fare', 'Cabin'])

# print ("Useless Columns Dropped! \n New Columns are")
# print (trainSet.columns)


trainSet.loc[trainSet['Sex'] == 'male', 'Sex'] = 1
trainSet.loc[trainSet['Sex'] == 'female', 'Sex'] = 0

trainSet.loc[trainSet['Embarked'] == 'C', 'Embarked'] = 1
trainSet.loc[trainSet['Embarked'] == 'Q', 'Embarked'] = 2
trainSet.loc[trainSet['Embarked'] == 'S', 'Embarked'] = 3

# print(trainSet.shape)

# trainSet.dropna(axis = 0, inplace = True)
trainSet['Embarked'] = pd.to_numeric(trainSet['Embarked'], errors='coerce')
print (trainSet.dtypes)
trainSet.interpolate(axis = 1, inplace = True)


# print(trainSet.shape)

# print(trainSet.head)

outpath = Path('~/Documents/Kaggle Projects/Titanic Project/out.csv') 


# trainSet.to_csv(outpath, index = False)

X = trainSet.loc[:, ['Pclass','Sex', 'Age', 'SibSp', 'Parch','Embarked']].to_numpy()
Y = trainSet.loc[:, 'Survived'].to_numpy()

# print (X)
# print (Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


LogModel = LogisticRegression()
LogModel.fit(X_train,Y_train)

Y_pred=LogModel.predict(X_test)

cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
print(cnf_matrix)

# print ("Model Parameters, ", LogModel.get_params())

print (LogModel.score(X_test, Y_test, sample_weight=None))


dataSet  = pd.read_csv("~/Documents/Kaggle Projects/Titanic Project/test.csv")
testSet = dataSet.drop(columns = ['Name', 'Ticket', 'Fare', 'Cabin'])


testSet.loc[testSet['Sex'] == 'male', 'Sex'] = 1
testSet.loc[testSet['Sex'] == 'female', 'Sex'] = 0

testSet.loc[testSet['Embarked'] == 'C', 'Embarked'] = 1
testSet.loc[testSet['Embarked'] == 'Q', 'Embarked'] = 2
testSet.loc[testSet['Embarked'] == 'S', 'Embarked'] = 3

# testSet.dropna(axis = 0, inplace = True)
testSet.interpolate(axis = 1, inplace = True)

print(testSet.head)

XT = testSet.loc[:, ['Pclass','Sex', 'Age', 'SibSp', 'Parch','Embarked']].to_numpy()

YT=LogModel.predict(XT)
Id = testSet['PassengerId'].to_numpy()



# YT =pd.DataFrame(YT, columns=['Survived'])

print('YT Shape', YT.shape)

# Submission = testSet.insert(1,'Survived',YT)
# Id = pd.DataFrame(testSet['PassengerId'], columns=['PassengerId'])
#Id = Id.reindex_like(YT)
print('Id Shape', Id.shape)
# Submission = pd.concat([Id, YT], axis=1)

# print(Submission.head)

# Sub = pd.DataFrame('PassengerID' : Id, 'Survived', YT)

# outpath = Path('~/Documents/Kaggle Projects/Titanic Project/Submission.csv') 

# Submission.to_csv(outpath, index = False)
subn = np.array([Id,YT]).T
# sub = pd.DataFrame(data = [(Id , YT)],columns=['Surv','Id'] )

print(subn)
np.savetxt("sub.csv", subn, delimiter=",")




