import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.train_data.columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']
        self.test_data = pd.read_csv(test_data_path)

    def predict(self):

        #Remove outliers
        self.train_data=self.train_data[self.train_data['quality']!=9]
        self.train_data=self.train_data[self.train_data['quality']!=3]
        
        X=self.train_data[['volatile acidity','free sulfur dioxide','alcohol']] #Feature selection        
        y=self.train_data['quality'].values
      
        
        #sc = StandardScaler()
        #X = sc.fit_transform(X)

        # Train the model        
        model = RandomForestClassifier(n_estimators = 200)
        model.fit(X, y)

        '''
        # predicting the results for the test set
        y_pred = model.predict(X)

        # calculating the training and testing accuracies
        print("Training accuracy :", model.score(X, y))
  
        # classification report
        print(classification_report(y, y_pred))

        # confusion matrix
        print(confusion_matrix(y, y_pred))

        model_eval = cross_val_score(estimator = model, X = X, y = y, cv = 10)
        print('mean ', model_eval.mean())
        '''
        # Predict on test set and save the prediction
        self.test_data=self.test_data[['volatile acidity','free sulfur dioxide','alcohol']]
        submission = model.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)


submission1 = Submission('train.csv','test.csv')
print("Submission ready")
submission1.predict()
