"""
    ex3_1

    written by : Dvir Gantz 300852431
                 Darina Poyarkov 316775402

"""
"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
"""
print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# #############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

N = []
durations = []
precisions = []
logistic_precision = []

# Load Data
X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, 'float32')
X, Y = nudge_dataset(X, y)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

for i in range (2, 21):
    N.append(i**2)

    # #############################################################################
    # Training
    
    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = i**2
    logistic.C = 6000
    
    before = time.clock()
    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, Y_train)
    after = time.clock()
#    print ("I = ", i)
#    print ("Time = ", after-before)
    
    durations.append(after-before)
    
    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.
    raw_pixel_classifier.fit(X_train, Y_train)
    
    # #############################################################################
    # Evaluation
    
    Y_pred = rbm_features_classifier.predict(X_test)
    
    precisions.append(metrics.precision_score(Y_test, Y_pred, average='weighted'))
#    print("Logistic regression using RBM features:\n%s\n" % (
#        metrics.classification_report(Y_test, Y_pred)))
    
    Y_pred = raw_pixel_classifier.predict(X_test)
    logistic_precision.append(metrics.precision_score(Y_test, Y_pred, average='weighted'))
#    print("Logistic regression using raw pixel features:\n%s\n" % (
#        metrics.classification_report(Y_test, Y_pred)))
    
    # #############################################################################
    # Plotting
#    plt.figure(i)
#    plt.figure(figsize=(4.2, 4))
#    for j, comp in enumerate(rbm.components_):
#        plt.subplot(i, i, j + 1)
#        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
#                   interpolation='nearest')
#        plt.xticks(())
#        plt.yticks(())
#    plt.suptitle('{0} components extracted by RBM'.format(i**2), fontsize=16)
#    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#    
#    plt.savefig('fix_{0}'.format(i))

plt.figure(20)
plt.plot(N,durations)
plt.xlabel("N")
plt.ylabel('Value')
plt.title('Fit measurements')
#plt.savefig('Fit measurements')

plt.figure(21)
plt.plot(N,precisions)
plt.plot(N,logistic_precision)
plt.xlabel("N")
plt.ylabel('Value')
plt.title('AVG Precisions')
#plt.savefig('AVG Precisions')
plt.show()

print("X_train shape = ", np.shape(X_train))
print("X_test shape = ",np.shape(X_test))
print("X_train transform shape = ",np.shape(rbm.transform(X_train)))
print("rbm.intercept_hidden_ shape = ",np.shape(rbm.intercept_hidden_))