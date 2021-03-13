import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.utils import resample
  
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn.metrics import fbeta_score, make_scorer,f1_score



class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.train_data.columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']
        self.test_data = pd.read_csv(test_data_path)

    def predict(self):


        df=self.train_data.copy()

        '''# Separate majority and minority classes
        df_majority = df[df.quality==7]
        df_minority1 = df[df.quality==9]
         
        # Upsample minority class
        df_minority_upsampled1 = resample(df_minority1, 
                                         replace=True,     # sample with replacement
                                         n_samples=704,    # to match majority class
                                         random_state=123) # reproducible results

        #==========================
        df_minority2 = df[df.quality==3]
         
        # Upsample minority class
        df_minority_upsampled2 = resample(df_minority2, 
                                         replace=True,     # sample with replacement
                                         n_samples=704,    # to match majority class
                                         random_state=123) # reproducible results
         
        #===============================
        df_minority3 = df[df.quality==4]
         # Upsample minority class
        df_minority_upsampled3 = resample(df_minority3, 
                                         replace=True,     # sample with replacement
                                         n_samples=704,    # to match majority class
                                         random_state=123) # reproducible results
         
        #===============================
        df_minority4 = df[df.quality==8]
         
        # Upsample minority class
        df_minority_upsampled4 = resample(df_minority4, 
                                         replace=True,     # sample with replacement
                                         n_samples=704,    # to match majority class
                                         random_state=123) # reproducible results
         
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled1,df_minority_upsampled2,df_minority_upsampled3,df_minority_upsampled4])

         
        # Display new class counts
        df_upsampled.quality.value_counts()


        df_majority1 = df[df.quality==6]
        df_minority = df[df.quality==7]
         
        # Downsample majority class
        df_majority_downsampled1 = resample(df_majority1, 
                                         replace=False,    # sample without replacement
                                         n_samples=704,     # to match minority class
                                         random_state=123) # reproducible results
        df_majority2 = df[df.quality==5]                                 
        # Downsample majority class
        df_majority_downsampled2 = resample(df_majority2, 
                                         replace=False,    # sample without replacement
                                         n_samples=704,     # to match minority class
                                         random_state=123) # reproducible results                                 
         
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled1,df_majority_downsampled2, df_minority])
         
        df_downsampled=df_downsampled[df_downsampled['quality'] != 7]
        # Display new class counts
        df_downsampled.quality.value_counts()


        df_balanced= pd.concat([df_downsampled,df_upsampled])
        df_balanced.quality.value_counts()
        '''



        #Remove outliers
        self.train_data=self.train_data[self.train_data['quality']!=9]
        #self.train_data=self.train_data[self.train_data['quality']!=3]
        
        #self.train_data=self.train_data[self.train_data['quality']!=8]
        #self.train_data=self.train_data[self.train_data['quality']!=4]
        

        #pca = PCA(n_components=4)
        #self.train_data=np.array(pca.fit_transform(self.train_data))
        #self.test_data=np.array(pca.fit_transform(self.test_data))
        
        X,y = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.25, random_state = 2489)


        #X = X[interested_features].values
    
        '''sc = StandardScaler()
        X = sc.fit_transform(X)
        '''

        #print(scores)

        # Train the model        
        model = RandomForestClassifier(n_estimators = 1000,class_weight='balanced',random_state=1)
        #model = ExtraTreesClassifier(n_estimators = 5000,class_weight='balanced')
        
        #model = SVC(C = 1.3, gamma =  1.3, kernel= 'rbf')
        model.fit(X_train, y_train)

        # predicting the results for the test set
        y_pred = model.predict(X_test)

        
        # calculating the training and testing accuracies
        print("Training accuracy :", model.score(X_train, y_train))
  
        # classification report
        print(classification_report(y_test, y_pred))

        # confusion matrix
        print(confusion_matrix(y_test, y_pred))

        model_eval = cross_val_score(estimator = model, X = X, y = y, cv = 10)
        print('mean ', model_eval.mean())
        print("F-score on the testing data:",f1_score(y_pred, y_test,average='micro'))

        # Predict on test set and save the prediction
        #self.test_data=self.test_data[]

        submission = model.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)



submission1 = Submission('train.csv','test.csv')
print("Submission ready")
submission1.predict()



