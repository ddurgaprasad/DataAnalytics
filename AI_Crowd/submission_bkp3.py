from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#LORAS
from sklearn.neighbors import NearestNeighbors
import concurrent.futures
import pandas as pd
import numpy as np

def knn(min_class_points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(min_class_points)
    _, indices = nbrs.kneighbors(min_class_points)
    neighbourhood = []
    for i in (indices):
        neighbourhood.append(min_class_points[i])
    return(np.asarray(neighbourhood)) 

def neighbourhood_oversampling(args):
    # Extracting arguments
    neighbourhood,k,num_shadow_points,list_sigma_f,num_generated_points,num_hyp_points,num_aff_comb,random_state = args
    # Setting seed
    np.random.seed(random_state)
    # Calculating shadow points
    neighbourhood_shadow_sample = []
    for i in range(k):
        q = neighbourhood[i]
        for _ in range(num_shadow_points):
            shadow_points = q + np.random.normal(0,list_sigma_f)
            neighbourhood_shadow_sample.append(shadow_points)
    # Selecting randomly num_aff_comb shadow points
    idx = np.random.randint(num_shadow_points*k, size=(num_generated_points,num_aff_comb))
    # Create random weights for selected points
    affine_weights = []
    for _ in range(num_hyp_points):
        # Create random weights for selected points
        weights = np.random.randint(100, size=idx.shape)
        sums = np.repeat(np.reshape(np.sum(weights,axis=1),(num_generated_points,1)), num_aff_comb, axis=1)
        # Normalise the weights
        affine_weights.append(np.divide(weights,sums))
    selected_points = np.array(neighbourhood_shadow_sample)[idx,:]
    # Performing dot product beteen points and weights
    neighbourhood_loras_set = []
    for affine_weight in affine_weights:
        generated_LoRAS_sample_points = list(np.dot(affine_weight, selected_points).diagonal().T)
        neighbourhood_loras_set.extend(generated_LoRAS_sample_points)
    
    return neighbourhood_loras_set

def loras_oversampling(min_class_points, k, num_shadow_points, list_sigma_f, num_generated_points, num_hyp_points, num_aff_comb, random_state):
    # Calculating neighbourhoods of each minority class parent data point p in min_class_points
    neighbourhoods = knn(min_class_points, k)
    # Preparing arguments
    args = []
    for neighbourhood in neighbourhoods:
        arg = (neighbourhood, k, num_shadow_points, list_sigma_f, num_generated_points, num_hyp_points, num_aff_comb, random_state)
        args.append(arg)
    # Generating points
    loras_set = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for neighbourhood_loras_set in executor.map(neighbourhood_oversampling, args):
            # Adding generated LoRAS points from specific neighbourhood
            loras_set.extend(neighbourhood_loras_set)
    
    return np.asarray(loras_set)

def fit_resample(maj_class_points, min_class_points, k=None, num_shadow_points=None, list_sigma_f=None, num_generated_points=None, num_hyp_points=1, num_aff_comb=None, random_state=42):
    
    # Verifying constraints
    if len(min_class_points)==0:
        print("[PARAMETER ERROR] Empty minority class")
        raise SystemExit
    if len(maj_class_points)==0:
        print("[PARAMETER ERROR] Empty majority class")
        raise SystemExit
    if len(min_class_points) > len(maj_class_points):
        print("[PARAMETER ERROR] Number of points in minority class exceeds number of points in the majority class")
        raise SystemExit
    
    # Completing missing parameters w/ default values
    if k is None:
        k = 5 if len(min_class_points)<100 else 30
    if num_aff_comb is None:
        num_aff_comb = min_class_points.shape[1]
    if num_shadow_points is None:
        import math
        num_shadow_points = max(math.ceil(2*num_aff_comb / k),40)
    if list_sigma_f is None:
        list_sigma_f = [.005]*min_class_points.shape[1]
    if not isinstance(list_sigma_f, list):
        list_sigma_f = [list_sigma_f]*min_class_points.shape[1]
    if num_generated_points is None:
        import math
        num_generated_points = math.ceil((len(maj_class_points) + len(min_class_points)) / (num_hyp_points * len(min_class_points)))
        
    # Verifying constraints
    if k <= 1:
        print("[PARAMETER ERROR] Value of parameter k is too small")
        raise SystemExit
    if k > len(min_class_points):
        print("[PARAMETER ERROR] Value of parameter k is too large for minority class points")
        raise SystemExit
    if num_shadow_points < 1:
        print("[PARAMETER ERROR] Number of shadow points is too small")
        raise SystemExit
    if not all(elem >= 0.0 and elem <= 1.0 for elem in list_sigma_f):
        print("[PARAMETER ERROR] All elements in list of sigmas have to be in [0.0,1.0]")
    if num_aff_comb < 1:
        print("[PARAMETER ERROR] Number of affine combinations is too small")
        raise SystemExit
    if num_aff_comb > k * num_shadow_points:
        print("[PARAMETER ERROR] Number of affine combinations must be smaller or equal to k * number of shadow points")
        raise SystemExit
    
    min_class_points_copy = np.copy(min_class_points)

    return np.concatenate((min_class_points,loras_oversampling(min_class_points_copy, k, num_shadow_points, list_sigma_f, num_generated_points, num_hyp_points, num_aff_comb, random_state)))


import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from collections import Counter

__author__ = 'stavrianos'

# Link to paper:  bit.ly/22KgAnP


class ADASYN(object):
    """
    Oversampling parent class with the main methods required by scikit-learn:
    fit, transform and fit_transform
    """

    def __init__(self,
                 ratio=0.5,
                 imb_threshold=0.5,
                 k=5,
                 random_state=None,
                 verbose=True):
        """
        :ratio:
            Growth percentage with respect to initial minority
            class size. For example if ratio=0.65 then after
            resampling minority class(es) will have 1.65 times
            its initial size
        :imb_threshold:
            The imbalance ratio threshold to allow/deny oversampling.
            For example if imb_threshold=0.5 then minority class needs
            to be at most half the size of the majority in order for
            resampling to apply
        :k:
            Number of K-nearest-neighbors
        :random_state:
            seed for random number generation
        :verbose:
            Determines if messages will be printed to terminal or not
        Extra Instance variables:
        :self.X:
            Feature matrix to be oversampled
        :self.y:
            Class labels for data
        :self.clstats:
            Class populations to determine minority/majority
        :self.unique_classes_:
            Number of unique classes
        :self.maj_class_:
            Label of majority class
        :self.random_state_:
            Seed
        """

        self.ratio = ratio
        self.imb_threshold = imb_threshold
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.clstats = {}
        self.num_new = 0
        self.index_new = []


    def fit(self, X, y):
        """
        Class method to define class populations and store them as instance
        variables. Also stores majority class label
        """
        self.X = check_array(X)
        self.y = np.array(y).astype(np.int64)
        self.random_state_ = check_random_state(self.random_state)
        self.unique_classes_ = set(self.y)

        # Initialize all class populations with zero
        for element in self.unique_classes_:
            self.clstats[element] = 0

        # Count occurences of each class
        for element in self.y:
            self.clstats[element] += 1

        # Find majority class
        v = list(self.clstats.values())
        k = list(self.clstats.keys())
        self.maj_class_ = k[v.index(max(v))]

        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, len(self.unique_classes_)))

    def transform(self, X, y):
        """
        Applies oversampling transformation to data as proposed by
        the ADASYN algorithm. Returns oversampled X,y
        """
        self.new_X, self.new_y = self.oversample()

    def fit_transform(self, X, y):
        """
        Fits the data and then returns the transformed version
        """
        self.fit(X, y)
        self.new_X, self.new_y = self.oversample()

        self.new_X = np.concatenate((self.new_X, self.X), axis=0)
        self.new_y = np.concatenate((self.new_y, self.y), axis=0)

        return self.new_X, self.new_y

    def generate_samples(self, x, knns, knnLabels, cl):

        # List to store synthetically generated samples and their labels
        new_data = []
        new_labels = []
        for ind, elem in enumerate(x):
            # calculating k-neighbors that belong to minority (their indexes in x)
            # Unfortunately knn returns the example itself as a neighbor. So it needs
            # to be ignored thats why it is iterated [1:-1] and
            # knnLabelsp[ind][+1].
            min_knns = [ele for index,ele in enumerate(knns[ind][1:-1])
                         if knnLabels[ind][index +1] == cl]

            if not min_knns:
                continue

            # generate gi synthetic examples for every minority example
            for i in range(0, int(self.gi[ind])):
                # randi holds an integer to choose a random minority kNNs
                randi = self.random_state_.random_integers(
                    0, len(min_knns) - 1)
                # l is a random number in [0,1)
                l = self.random_state_.random_sample()
                # X[min_knns[randi]] is the Xzi on equation [5]
                si = self.X[elem] + \
                    (self.X[min_knns[randi]] - self.X[elem]) * l
                    
                new_data.append(si)
                new_labels.append(self.y[elem])
                self.num_new += 1

        return(np.asarray(new_data), np.asarray(new_labels))

    def oversample(self):
        """
        Preliminary calculations before generation of
        synthetic samples. Calculates and stores as instance
        variables: img_degree(d),G,ri,gi as defined by equations
        [1],[2],[3],[4] in the original paper
        """

        try:
            # Checking if variable exists, i.e. if fit() was called
            self.unique_classes_ = self.unique_classes_
        except:
            raise RuntimeError("You need to fit() before applying tranform(),"
                               "or simply fit_transform()")
        int_X = np.zeros([1, self.X.shape[1]])
        int_y = np.zeros([1])
        # Iterating through all minority classes to determine
        # if they should be oversampled and to what extent
        for cl in self.unique_classes_:
            # Calculate imbalance degree and compare to threshold
            imb_degree = float(self.clstats[cl]) / \
                self.clstats[self.maj_class_]
            if imb_degree > self.imb_threshold:
                if self.verbose:
                    print('Class %s is within imbalance threshold' % cl)
            else:
                # G is the number of synthetic examples to be synthetically
                # produced for the current minority class
                self.G = (self.clstats[self.maj_class_] - self.clstats[cl]) \
                            * self.ratio

                # ADASYN is built upon eucliden distance so p=2 default
                self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k + 1)
                self.nearest_neighbors_.fit(self.X)

                # keeping indexes of minority examples
                minx = [ind for ind, exam in enumerate(self.X) if self.y[ind] == cl]

                # Computing kNearestNeighbors for every minority example
                knn = self.nearest_neighbors_.kneighbors(
                    self.X[minx], return_distance=False)

                # Getting labels of k-neighbors of each example to determine how many of them
                # are of different class than the one being oversampled
                knnLabels = self.y[knn.ravel()].reshape(knn.shape)

                tempdi = [Counter(i) for i in knnLabels]

                # Calculating ri as defined in ADASYN paper:
                # No. of k-neighbors belonging to different class than the minority divided by K
                # which is ratio of friendly/non-friendly neighbors
                self.ri = np.array(
                    [(sum(i.values())- i[cl]) / float(self.k) for i in tempdi])

                # Normalizing so that ri is a density distribution (i.e.
                # sum(ri)=1)
                if np.sum(self.ri):
                    self.ri = self.ri / np.sum(self.ri)

                # Calculating #synthetic_examples that need to be generated for
                # each minority instance and rounding to nearest integer because
                # it can't produce e.g 2.35 new examples.
                self.gi = np.rint(self.ri * self.G)

                # Generation of synthetic samples
                inter_X, inter_y = self.generate_samples(
                                     minx, knn, knnLabels, cl)
                # in case no samples where generated at all concatenation
                # won't be attempted
                if len(inter_X):
                    int_X = np.concatenate((int_X, inter_X), axis=0)
                if len(inter_y):
                    int_y = np.concatenate((int_y, inter_y), axis=0)
        # New samples are concatenated in the beggining of the X,y arrays
        # index_new contains the indiced of artificial examples
        self.index_new = [i for i in range(0,self.num_new)]
        return(int_X[1:-1], int_y[1:-1])

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
   
       
        X,y = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]

 

        adsn = ADASYN(k=7,imb_threshold=0.6, ratio=0.75)
        X, y = adsn.fit_transform(X,y)  # your imbalanced dataset is in X,y


        #X = X[interested_features].values
    
        #sc = StandardScaler()
        #X = sc.fit_transform(X)

        # Train the model        
        model = RandomForestClassifier(n_estimators = 200,random_state=2489)
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
        #self.test_data=self.test_data[]
        submission = model.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)


#submission1 = Submission('train.csv','test.csv')
#print("Submission ready")
#submission1.predict()
