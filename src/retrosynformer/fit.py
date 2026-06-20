import math
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt


df = pd.DataFrame(list(range(1, 151)), columns=['epoch'])
bias, sigma = .5, .5
df['y'] = df['epoch'].apply(np.log) + sigma * np.random.randn(len(df)) + bias
df = df.set_index('epoch')

p = PolynomialFeatures(degree=2)
df = pd.concat([
    df, 
    pd.DataFrame(
        p.fit_transform(
            df[[list(df.columns)[0]]]),
        columns=[f'x_{i}' for i in range(p.degree+int(p.include_bias))],
        index=df.index)
    ], axis=1)
X = df[[c for c in df.columns if c.startswith('x_')]]
Y = df[[c for c in df.columns if c.startswith('y')]]

m = Ridge()
m.fit(X, Y)
df['y_pred'] = m.predict(X)
df[[c for c in df.columns if c.startswith('y')]].plot()
plt.grid('on')
plt.show()

