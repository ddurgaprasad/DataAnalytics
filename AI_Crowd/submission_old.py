import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# import os
# cmd = "pip install imbalanced-learn"
# returned_value = os.system(cmd)  # returns the exit code in unix
# print('returned value:', returned_value)

import imblearn
from imblearn.over_sampling import SMOTE

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.test_data = pd.read_csv(test_data_path,header=None)
        self.test_data.drop(0,inplace=True)

    def predict(self):

        X,y = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]
		
        sc = StandardScaler()
        X = sc.fit_transform(X)
                        
        # transform the dataset
        oversample=SMOTE(sampling_strategy='auto',k_neighbors=2)
        X_SMOTE, y_SMOTE = oversample.fit_resample(X, y)
		
        # Train the model
        # creating the model
        model = RandomForestClassifier(n_estimators = 200,n_jobs=-1,verbose=2)
        model.fit(X_SMOTE, y_SMOTE)

        # Predict on test set and save the prediction
        submission = model.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)
		

submission1 = Submission('train.csv','test.csv')
print("submission1")
submission1.predict()
