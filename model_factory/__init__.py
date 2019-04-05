
from sklearn.neighbors.kde import KernelDensity
import numpy as np
X = np.array([ 0.38982287,0.14883478,0.20893729,0.55553359,0.34115332,0.3279272
,0.63697904,0.91232055,0.64658582,0.85961515])


ret = np.histogram([1, 2, 1], normed=True)
print(ret)
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X.reshape(-1,1))
# ret = kde.score_samples(np.array([0.461516 ]).reshape(-1,1))
# print(ret, np.exp(ret))
#
#
#
# sns.distplot(X, hist=True)
# plt.show()