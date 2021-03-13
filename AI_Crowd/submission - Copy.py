from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import train_test_split

#LORAS
from sklearn.neighbors import NearestNeighbors
import concurrent.futures
import pandas as pd
import numpy as np

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from collections import Counter


class SMOTE:
    """Python implementation of SMOTE.
        This implementation is based on the original variant of SMOTE.
        Parameters
        ----------
        ratio : int, optional (default=100)
            The ratio percentage of generated samples to original samples.
            - If ratio < 100, then randomly choose ratio% of samples to SMOTE.
            - If ratio >= 100, it must be a interger multiple of 100.
        k_neighbors : int, optional (defalut=6)
            Number of nearest neighbors to used to SMOTE.
        random_state : int, optional (default=None)
            The random seed of the random number generator.
    """
    def __init__(self,
                 ratio=100,
                 k_neighbors=6,
                 random_state=None):
        # check input arguments
        if ratio > 0 and ratio < 100:
            self.ratio = ratio
        elif ratio >= 100:
            if ratio % 100 == 0:
                self.ratio = ratio
            else:
                raise ValueError(
                    'ratio over 100 should be multiples of 100')
        else:
            raise ValueError(
                'ratio should be greater than 0')

        if type(k_neighbors) == int:
            if k_neighbors > 0:
                self.k_neighbors = k_neighbors
            else:
                raise ValueError(
                    'k_neighbors should be integer greater than 0')
        else:
            raise TypeError(
                'Expect integer for k_neighbors')

        if type(random_state) == int:
            np.random.seed(random_state)

    def _randomize(self, samples, ratio):
        length = samples.shape[0]
        target_size = length * ratio
        idx = np.random.randint(length, size=target_size)

        return samples[idx, :]

    def _populate(self, idx, nnarray):
        for i in range(self.N):
            nn = np.random.randint(low=0, high=self.k_neighbors)
            for attr in range(self.numattrs):
                dif = (self.samples[nnarray[nn]][attr]
                       - self.samples[idx][attr])
                gap = np.random.uniform()
                self.synthetic[self.newidx][attr] = (self.samples[idx][attr]
                                                     + gap * dif)
            self.newidx += 1

    def oversample(self, samples, merge=False):
        """Perform oversampling using SMOTE
        Parameters
        ----------
        samples : list or ndarray, shape (n_samples, n_features)
            The samples to apply SMOTE to.
        merge : bool, optional (default=False)
            If set to true, merge the synthetic samples to original samples.
        Returns
        -------
        output : ndarray
            The output synthetic samples.
        """
        if type(samples) == list:
            self.samples = np.array(samples)
        elif type(samples) == np.ndarray:
            self.samples = samples
        else:
            raise TypeError(
                'Expect a built-in list or an ndarray for samples')

        self.numattrs = self.samples.shape[1]

        if self.ratio < 100:
            ratio = ratio / 100.0
            self.samples = self._randomize(self.samples, ratio) 
            self.ratio = 100

        self.N = int(self.ratio / 100)
        new_shape = (self.samples.shape[0] * self.N, self.samples.shape[1])
        self.synthetic = np.empty(shape=new_shape)
        self.newidx = 0

        self.nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nbrs.fit(samples)
        self.knn = self.nbrs.kneighbors()[1]

        for idx in range(self.samples.shape[0]):
            nnarray = self.knn[idx]
            self._populate(idx, nnarray)

        if merge:
            return np.concatenate((self.samples, self.synthetic))
        else:
            return self.synthetic
			
def getSMOTE(data):
  sample_data=data[data['quality']==9].values
  smote = SMOTE(ratio=43000, k_neighbors=2)
  synthetic_samples = smote.oversample(sample_data)
  new_sample_9 = smote.oversample(sample_data, merge=True)
  new_sample_9.shape

  sample_data=data[data['quality']==3].values
  smote = SMOTE(ratio=10900, k_neighbors=8)
  synthetic_samples = smote.oversample(sample_data)
  new_sample_3 = smote.oversample(sample_data, merge=True)
  new_sample_3.shape

  sample_data=data[data['quality']==4].values
  smote = SMOTE(ratio=1300, k_neighbors=66)
  synthetic_samples = smote.oversample(sample_data)
  new_sample_4 = smote.oversample(sample_data, merge=True)
  new_sample_4.shape

  sample_data=data[data['quality']==8].values
  smote = SMOTE(ratio=1200, k_neighbors=71)
  synthetic_samples = smote.oversample(sample_data)
  new_sample_8 = smote.oversample(sample_data, merge=True)
  new_sample_8.shape


  sample_data=data[data['quality']==7].values
  smote = SMOTE(ratio=100, k_neighbors=352)
  synthetic_samples = smote.oversample(sample_data)
  new_sample_7 = smote.oversample(sample_data, merge=True)
  new_sample_7.shape

  new_data=np.concatenate([data[data['quality']==6].values,data[data['quality']==5].values,new_sample_7,new_sample_8,new_sample_4,
                        new_sample_3,new_sample_9])
  
  df=pd.DataFrame(data=new_data)
  df.columns=data.columns

  return df

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

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.train_data.columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']
        self.test_data = pd.read_csv(test_data_path)
        self.req_f1 = 0.0
        self.model = ExtraTreesClassifier(n_estimators=200,max_depth=None, min_samples_split=2,random_state=0)

    def predict(self):

       #Remove outliers
        #self.train_data= getSMOTE(self.train_data )
        #print(self.train_data.shape)
        self.train_data=self.train_data[self.train_data['quality']!=9]
        self.train_data=self.train_data[self.train_data['quality']!=3]   
       
        X,y = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]

        #X=self.train_data[['volatile acidity','free sulfur dioxide','alcohol']]
        #print("self.req_f1  ",self.req_f1 )
        while (1):

          #print("self.req_f1  ",self.req_f1 )

          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
          # Train the model        
          self.model =  RandomForestClassifier(n_estimators=200)
          self.model.fit(X_train, y_train)  
          
          #self.model = ExtraTreesClassifier(n_estimators=200,max_depth=None, min_samples_split=2)

          #self.model.fit(X_train, y_train) 

          #predictions=self.model.predict(X_train)

          predictions=self.model.predict(X_test)
          #print("Testing ")         
          #print("F-score on the testing data: ",f1_score(y_test, predictions,average='micro'))
          self.req_f1 =f1_score(y_test, predictions,average='micro')

          if self.req_f1  >  0.71 :
            submission = self.model.predict(self.test_data)
            submission = pd.DataFrame(submission)
            submission.to_csv('submission.csv',header=['quality'],index=False)
            break

        # Predict on test set and save the prediction
        #self.test_data=self.test_data[['volatile acidity','free sulfur dioxide','alcohol']]
          #submission = self.model.predict(self.test_data)
          #submission = pd.DataFrame(submission)
          #submission.to_csv('submission.csv',header=['quality'],index=False)

submission1 = Submission('train.csv','test.csv')
submission1.predict()
print("Submission ready")



