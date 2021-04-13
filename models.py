import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import confusion_matrix

np.random.seed(12345)

data = pd.read_excel('completedData.xlsx')
X = data[['Following', 'Followers', 'Tweets', 'avgmention', 'perlinktweets', 'persimilartweet']]
y = data['Bot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Parameter Space
RF_n_estimators = (100,250,500)
RF_min_samples_split = (2,3,4,5)
RF_params = [ {'n_estimators' : ne, 'min_samples_split' : mss} \
              for ne, mss in product(RF_n_estimators, RF_min_samples_split) ]

GB_n_estimators = (100,250,500)
GB_learning_rate = (0.05, 0.1, 0.2)
GB_params = [ {'n_estimators' : ne, 'learning_rate' : lr} \
              for ne, lr in product(GB_n_estimators, GB_learning_rate) ]

KNN_n_neighbors = (5,10,15,20)
KNN_params = [ {'n_neighbors' : k} for k in KNN_n_neighbors ]

LM_C = (0.1, 0.5, 1.0)
LM_params = [ {'C' : c} \
              for c in LM_C ]

# Parameter Search
models = [RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), LogisticRegression()]
param_lists = [RF_params, GB_params, KNN_params, LM_params]
train_results = []

print("Parameter Search")
for model, params in zip(models, param_lists):
    param_scores = []
    print("Checking " + str(type(model)))
    for param in params:
        print("\t Parameters: " + str(param))
        model.set_params(**param)
        cv_scores = cross_val_score(model, X_train, y_train, cv = 5)
        mean_cv = np.mean(cv_scores)
        param_scores.append(mean_cv)
        print("\t\t Mean CV score: " + str(mean_cv))


    max_score = np.max(param_scores)
    max_score_index = np.where(param_scores == max_score)[0][0]
    best_params = params[max_score_index]

    print("\t\t\t Best params: " + str(best_params))
    print("\t\t\t Score:" + str(max_score))
    train_results.append( (model, best_params, max_score) )

best_model, best_params, _ = max(train_results, key = lambda x: x[2])
best_model.fit(X_train, y_train)
best_model.score(X_test, y_test)

print("Validation:")
for model, params, _ in train_results:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predict = model.predict(X_test)
    auc = roc_auc_score(y_test, predict)
    print("Model: {}\n \tMisclassification: {}\n \tAUC: {}\n".format(str(model), 1 - score, auc))




