# from gensim.models import Word2Vec
# sequence = [
#     ['123', '56'],
#     ['11', '55'],
#     ['22', '55']
# ]
# model = Word2Vec(sequence, size=100, window=5, min_count=1, workers=4)
# ret = model.predict_output_word(['11'])
# print(ret)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot

XX = np.array([[20,20]])
# probability of each component
print(clf.predict_proba(XX))
# log probability of samples
Z = clf.score_samples(XX)
# probabilities of samples
p = np.exp(Z)
