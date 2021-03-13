import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer,accuracy_score,f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import LinearSVC

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import imblearn
from imblearn.over_sampling import SMOTE


TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
#TRAIN_DATA_PATH = "/home/zestiot-15/Downloads/light_train.csv"#os.getenv("TRAIN_DATA_PATH")
#TEST_DATA_PATH = "/home/zestiot-15/Downloads/light_test.csv"#os.getenv("TEST_DATA_PATH")

# Prepare the training data
train_data = pd.read_csv(TRAIN_DATA_PATH)
         
#while(1):
X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]

#print(X_train.shape)
# transform the dataset
#oversample = SMOTE()
#X_train, y_train = oversample.fit_resample(X_train, y_train)
#print(X_train.shape)

#X_train, X_val, y_train, y_val = train_test_split(XX, yy, test_size = 0.2)
#project 1 SVC gamma = 0.05, C=3.7 => 0.680, 5.2,4.8 =? 0.681
#project 1, randomforest ==>{'criterion': 'gini', 'max_depth': 4, 'max_features': 'auto', 'n_estimators': 400}
#Project 2, randomforest ==>{'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'n_estimators': 500}
#Project 3, randomforest ==>{'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'n_estimators': 900}


#classifier = SVC(C=1,kernel='linear')#RandomForestClassifier(n_estimators=800)#SVC(break_ties=False, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.08, kernel='rbf', probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
classifier = RandomForestClassifier(criterion='gini',max_depth=40,max_features='log2',n_estimators=700)#SVC(break_ties=False, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.08, kernel='rbf', probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

classifier.fit(X_train, y_train)  

test_data = pd.read_csv(TEST_DATA_PATH)
submission = classifier.predict(test_data)
submission = pd.DataFrame(submission)
submission.to_csv('submission.csv', header=['class'], index=False) 
