import math
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt


df = pd.DataFrame(list(range(1, 151)), columns=['epoch'])
x_bias, x_sigma = .05, .05
y_bias, y_sigma = .3, .3
df['y_meas'] = (
        df['epoch'] + x_sigma * np.random.randn(len(df)) + x_bias 
    ).apply(np.log) + y_sigma * np.random.randn(len(df)) + y_bias

df = df.set_index('epoch', drop=False)

poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(df[['epoch']])
X = pd.DataFrame(X,
        columns=[f'x_{i}' for i in range(poly.degree+int(poly.include_bias))],
        index=df.index)
Y = df[[c for c in df.columns if c.startswith('y_')]]

df = pd.concat([X, Y], axis=1)

m = Ridge()
m.fit(X, Y)
df['y_pred'] = m.predict(X)
# df[[c for c in df.columns if c.startswith('y')]].plot()
plt.grid('on')
plt.show()

