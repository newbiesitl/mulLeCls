import sklearn.metrics
import numpy as np

y = [1,3,5,3,1]
y_hat = [1,60,60,60,1]

mae = sklearn.metrics.mean_absolute_error(y, y_hat)
mse = sklearn.metrics.mean_squared_error(y, y_hat)
rmse = np.sqrt(mse)
print(mae, mse, rmse)