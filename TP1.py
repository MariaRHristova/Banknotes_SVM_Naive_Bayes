# -*- coding: utf-8 -*-

#Created on Mon Oct 21 02:05:39 2019

# Importing the data 

import numpy as np 

data_train = np.loadtxt('D:\TP1_train.tsv', delimiter='\t')
data_test = np.loadtxt("D:\TP1_test.tsv", delimiter="\t")

# Shuffling the data 

#from sklearn.utils import shuffle   # > random shuffle
#data_train = shuffle(data_train)
#data_test = shuffle(data_test)

ranks_train = np.arange(data_train.shape[0])
np.random.shuffle(ranks_train)
data_train = data_train[ranks_train,:]

ranks_test = np.arange(data_test.shape[0])
np.random.shuffle(ranks_test)
data_test = data_test[ranks_test,:]

# Standartazing the data

means = np.mean(data_train[:,:-1],axis=0) # Xs train 
stdevs = np.std(data_train[:,:-1],axis=0) # Xs train 

data_train[:,:-1]= (data_train[:,:-1]-means)/stdevs
data_test[:,:-1] = (data_test[:,:-1]-means)/stdevs # we use the same scaling operators


# Kernel density ## Do we have to find the optimal bw now or after we divide them by class
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
#np.linspace(0.02, 0.6, 30)
params = {'bandwidth':   np.linspace(0.02, 0.6, 30) }
grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False,return_train_score= True)
grid.fit(data_train[:,:-1])  # Xs train 
grid.cv_results_['mean_test_score']
grid.cv_results_['mean_train_score']
print("Best CV parameter is", grid.best_params_)

import matplotlib.pyplot as plt
b1 = plt.plot(np.linspace(0.2, 6.01, 30), grid.cv_results_['mean_test_score'])
b2 = plt.plot(np.linspace(0.2, 6.01, 30), grid.cv_results_['mean_train_score'])
plt.title('Mean train and test scores for the bandwith parameter ')
plt.ylabel('Scores')
plt.xlabel('Bandwidth value')
plt.legend((b1[0], b2[0]), ('Test_score', 'Train_score'))
plt.savefig('NB.png', bbox_inches = 'tight', dpi = 600)

########################################
### Naive Bayes Classifier with KDE  ###
########################################

# 1. Subset by label
# For each class, find the log probability distribution of features 

data_train_1 = data_train[(data_train[:,4]==1)]
data_train_0 = data_train[(data_train[:,4]==0)]

feat_1 = data_train_1[:,0]

kde_1_feat_1 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_1_feat_1.fit(data_train_1[:,0].reshape(-1, 1))
logprob_1_feat_1 = kde_1_feat_1.score_samples(data_train_1[:,0].reshape(-1, 1))

feat_2 = data_train_1[:,1]

kde_1_feat_2 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_1_feat_2.fit(data_train_1[:,1].reshape(-1, 1))
logprob_1_feat_2 = kde_1_feat_2.score_samples(data_train_1[:,1].reshape(-1, 1))

feat_3 = data_train_1[:,2]

kde_1_feat_3 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_1_feat_3.fit(data_train_1[:,2].reshape(-1, 1))
logprob_1_feat_3 = kde_1_feat_3.score_samples(data_train_1[:,2].reshape(-1, 1))

feat_4 = data_train_1[:,3]

kde_1_feat_4 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_1_feat_4.fit(data_train_1[:,3].reshape(-1, 1))
logprob_1_feat_4 = kde_1_feat_4.score_samples(data_train_1[:,3].reshape(-1, 1))

sum_of_log_prob1 = (logprob_1_feat_1 + logprob_1_feat_2 + logprob_1_feat_3 + logprob_1_feat_4)
print(sum_of_log_prob1)

# the same for class 0 
kde_0_feat_1 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_0_feat_1.fit(data_train_0[:,0].reshape(-1, 1))
logprob_0_feat_1 = kde_1_feat_1.score_samples(data_train_0[:,0].reshape(-1, 1))

kde_0_feat_2 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_0_feat_2.fit(data_train_0[:,1].reshape(-1, 1))
logprob_0_feat_2 = kde_1_feat_2.score_samples(data_train_0[:,1].reshape(-1, 1))

kde_0_feat_3 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_0_feat_3.fit(data_train_0[:,2].reshape(-1, 1))
logprob_0_feat_3 = kde_1_feat_3.score_samples(data_train_0[:,2].reshape(-1, 1))

kde_0_feat_4 = KernelDensity(kernel='gaussian', bandwidth=0.14)
kde_0_feat_4.fit(data_train_0[:,3].reshape(-1, 1))
logprob_0_feat_4 = kde_1_feat_4.score_samples(data_train_0[:,3].reshape(-1, 1))

sum_of_log_prob0 = (logprob_0_feat_1 + logprob_0_feat_2 + logprob_0_feat_3 + logprob_0_feat_4)
print(sum_of_log_prob0)
# 2. For each class, find the log of the prior probability
# To classify:

np.log(data_train_1.shape[0]/data_train.shape[0])  #log( number of 1s in train / number in train)
np.log(data_train_0.shape[0]/data_train.shape[0])  # log(number of 1s in train / number in train)
logpriors =np.linspace (-0.6931471805599453, -0.6931471805599453)
logpriors[0,]

# 3. Find class for which the sum of the feature logs, plus the log of the prior, is greatest
a = sum_of_log_prob1 + logpriors[0,]
b = sum_of_log_prob0 + logpriors[1,]

# We have to compare them in order to classify new observations from the test test. 
     
########################################
#### Gaussian Naïve Bayes classifier ###
########################################

from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model_GNB = GaussianNB()

# Train the model using the training sets
model_GNB.fit(data_train[:,:-1],data_train[:,4])
data_train[:,:-1] 
data_train[:,4]
#Predict Output
y_pred_GNB = model_GNB.predict(data_test[:,:-1] )

# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("GNBC accuracy: ", metrics.accuracy_score(data_test[:,4], y_pred_GNB)*100)


#######################
### SVM Classifier  ###
#######################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

## Optimze gamma

# Instantiate an RBF SVM
model_SVM = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma': np.arange(0.2, 6.01, 0.2)} # change with marias code 
searcher = GridSearchCV(model_SVM, parameters, cv = 5, return_train_score= True, iid = False)
searcher.fit(data_train[:,:-1], data_train[:,4])
# Report the best parameters
print("Best CV params", searcher.best_params_)

# Plot 
import matplotlib.pyplot as plt
p1 = plt.plot(np.linspace(0.2, 6.01, 30), searcher.cv_results_['mean_test_score'])
p2 = plt.plot(np.linspace(0.2, 6.01, 30), searcher.cv_results_['mean_train_score'])
plt.title('Mean train and test scores for gamma parameter ')
plt.ylabel('Scores')
plt.xlabel('Gamma value')
plt.legend((p1[0], p2[0]), ('Test_score', 'Train_score'))
plt.savefig('SVM.png', dpi=300)
plt.close()

# Train the model 
SVM = SVC(C=1, kernel='rbf', gamma= 4 )

SVM.fit(data_train[:,:-1] ,data_train[:,4])

# Predict Output
y_pred_SVM = SVM.predict(data_test[:,:-1])

# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score: ",accuracy_score(y_pred_SVM, data_test[:,4])*100)

# Comparing Classifiers - McNemar test

#pip install mlxtend
from mlxtend.evaluate import mcnemar_table

tb = mcnemar_table(y_target = data_test[:,4], 
                   y_model1 = y_pred_SVM, 
                   y_model2 = y_pred_GNB)

print(tb)

chi_GNBC_SVM = ((abs(tb[0,1] - tb[1,0]) - 1)**2) / (tb[0,1] + tb[1,0])
print(chi_GNBC_SVM)

# Comparing Classifiers - Approximate normal test

# SVM test 
from sklearn.metrics import confusion_matrix
Conf_matrix_SVM = confusion_matrix(data_test[:,4], y_pred_SVM)
print(Conf_matrix_SVM)

X = (Conf_matrix_SVM[0,1] + Conf_matrix_SVM[1,0])
N = (np.sum(Conf_matrix_SVM))
p = X / N
print(p)

import math 
sigma = math.sqrt(N * p * (1-p))

low_conf = X - (1.96**sigma)
high_conf = X + (1.96**sigma)

print(" X_SVM belongs to this interval", low_conf, "to", high_conf)

# GNBC test

Conf_matrix_GNBC = confusion_matrix(data_test[:,4], y_pred_GNB)
print(Conf_matrix_GNBC)

X = (Conf_matrix_GNBC[0,1] + Conf_matrix_GNBC[1,0])
N = (np.sum(Conf_matrix_GNBC))
p = X / N
#print(p)

import math 
sigma = math.sqrt(N * p * (1-p))

low_conf = X - (1.96**sigma)
high_conf = X + (1.96**sigma)

print(" X_GNBC belongs to this interval", low_conf, "to", high_conf)







