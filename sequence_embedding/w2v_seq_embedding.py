

dat = [
    (101, 11, 'click'),
    (101, 10, 'click'),
    (101, 1, 'click'),
    (101, 2, 'click'),
    (111, 12, 'click'),
    (111, 11, 'click'),
    (111, 10, 'click'),
    (111, 1, 'click'),
    (112, 11, 'click'),
    (112, 10, 'click'),
    (113, 11, 'click'),
    (113, 10, 'click'),
]


user_interactions = {}
for each in dat:
    user_id, item_id, interaction_type = each
    if user_id in user_interactions:
        user_interactions[user_id].append(item_id)
    else:
        user_interactions[user_id] = [item_id]



total_interactions = []
for user_id in user_interactions:
    this_user_interaction = user_interactions[user_id]
    this_user_interaction = [str(x) for x in this_user_interaction]
    total_interactions.append(this_user_interaction)

print(total_interactions)
from gensim.models import Word2Vec

model = Word2Vec(total_interactions, size=5, window=100, min_count=1, workers=4)
ret = model.predict_output_word(['11'])
print(ret)
ret = model.similar_by_word('11')
print(ret)
ret = model.most_similar(['11'])
print(ret)



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
