from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold

from pandas import DataFrame

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def accuracy(model, X, y):
    kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
    results = cross_val_score(model, X, y, cv=kf)
    #Accuracy measure 
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))

print('======================================')
print('+++++++++++++ Cleaning +++++++++++++++') 
print('======================================')

#dropped other columns cause they seem irrelevant
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
X = train[features]
X = X[pd.notnull(X['Embarked'])]
y = X['Survived']
del X['Survived']

test = test[features[:-1]]
test = test[pd.notnull(test['Embarked'])]

# print(X.isnull().any())
# Age, cabin and embarked has empty values. Dropped cabin now as it has too many empty values

labelEncoderSex = LabelEncoder()
labelEncoderSex.fit(X['Sex'])
X['Sex'] = labelEncoderSex.transform(X['Sex'])
test['Sex'] = labelEncoderSex.transform(test['Sex'])

labelEncoderEmbarked = LabelEncoder()
labelEncoderEmbarked.fit(X['Embarked'])
X['Embarked'] = labelEncoderEmbarked.transform(X['Embarked'])
test['Embarked'] = labelEncoderEmbarked.transform(test['Embarked'])

X = X.fillna(X.mean())
test = test.fillna(test.mean())

print()
print('======================================')
print('+++++++++++++++ Model ++++++++++++++++') 
print('======================================')
print()
print('+++++++++ Hyper parameters +++++++++++')
print()

def bestParamCalculator(model, params, X, y):
    kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
    random = RandomizedSearchCV(estimator = model, param_distributions = params, cv = kf, verbose = 2, random_state = 42, n_jobs = -1, scoring='neg_mean_squared_error')
    random.fit(X, y)
    return random.best_estimator_

def randomForest(X, y):
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 200, num = 5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
    min_samples_split = [5, 10]

    random_grid = { 
            'n_estimators': n_estimators, 
            'max_features': max_features, 
            'max_depth': max_depth, 
            'min_samples_split': min_samples_split
    }

    model = bestParamCalculator(RandomForestClassifier(), random_grid, X, y)
    return model

def knn(X, y):
    random_grid = { 
                'n_neighbors': range(1,10), 
                'algorithm': ['auto', 'kd_tree'], 
                'weights' : ['distance', 'uniform']
    }

    bestModel = bestParamCalculator(KNeighborsClassifier(), random_grid, X, y)
    return bestModel

rf = randomForest(X, y)
# knn = knn(X, y)

accuracy(rf, X, y)
# accuracy(knn, X, y)
probability = rf.predict(test)

print()
print('************ CSV FILE ***************')
print()

idValues = test['PassengerId']
finalDataset = {
    'PassengerId' : idValues,
    'Survived' : probability
}

df = DataFrame(finalDataset, columns = ['PassengerId', 'Survived'])
df.to_csv (r'export.csv', index = None, header = True)