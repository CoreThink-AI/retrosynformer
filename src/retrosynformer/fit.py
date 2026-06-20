import math
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso

df = pd.DataFrame(list(range(1, 151)), columns=['epoch'])
df['y'] = df['epoch'].apply(np.log)
df = df.set_index('epoch')

from sklearn.preprocessing import PolynomialFeatures
p = PolynomialFeatures(df['epoch'])
p = PolynomialFeatures(df['epoch'], degree=2)
p = PolynomialFeatures?
p = PolynomialFeatures(degree=2, df['epoch'])
p = PolynomialFeatures(degree=2)
p.fit(df['epoch'])
p?
p.fit_transform(df['epoch'])
p.fit_transform(df[['epoch']])
p.fit_transform?
pd.DataFrame(p.fit_transform(df[['epoch']]))
pd.concat(df, pd.DataFrame(p.fit_transform(df[[list(df.columns)[0]]])))
pd.concat([df, pd.DataFrame(p.fit_transform(df[[list(df.columns)[0]]]))])
pd.concat([df, pd.DataFrame(p.fit_transform(df[[list(df.columns)[0]]]))], axis=1)
df = pd.concat([df, pd.DataFrame(p.fit_transform(df[[list(df.columns)[0]]]))], axis=1)
df = pd.concat([df, pd.DataFrame(p.fit_transform(df[[list(df.columns)[0]]]), columns=[f'x_{i}' for i in range(p.degree+int(p.include_bias))])], axis=1)
df
hist -f src/retrosynformer/fit.ipy
