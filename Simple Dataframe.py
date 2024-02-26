from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from io import StringIO


csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
"""
print(df)
print(df.isnull().sum()) # return dataframe with number of missing values, NaN, per column

print(df.dropna(axis=0)) # drop rows with missing values
print(df.dropna(axis=1)) # drop columns with at least on Nan in any row
print(df.dropna(how='all')) # only drop rows where all columns are NaN
print(df.dropna(thresh=4)) # drop rows that have fewer than 4 real values
print(df.dropna(subset=['C'])) # only drop rows where NaN appear in specific columns (here: C)
"""
# Simple Imputer (transformer API in scikit-learn, similar to the estimator API in scikit-learn for classifiers)
# impute missing values via the column mean, interpolation technique
imr = SimpleImputer(missing_values=np.nan, strategy='mean') # other strategies are median / most frequent (i.e. categorical feature values such as color names)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

# shorter, more convenient way to impute using mean() via pandas
df.fillna(df.mean())


# # Handling categorical data

# ## Nominal and ordinal features - Ordinal (t-shirt size XL > L > M), Nominal (t-shirt color, no order of colors)

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)


# Map ordinal features, for our learning algorithm to interpret correctly using integers.
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
print(df['size'])

# assuming the numerical difference is, we can transform our int values back to string representation
# Dictionary comprehension: {key_expression: value_expression for item in iterable}
inv_size_mapping = {v: k for k, v in size_mapping.items()} # new dict will have keys of size_mapping as its values and values of size_mapping as its keys
df['size'].map(inv_size_mapping)
#df['size'] = df['size'].map(inv_size_mapping)
print(df['size'])


# Encoding class labels

# create a mapping dict
# to convert class labels from strings to integers using enumerate, since class labels are non-ordinal
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
#print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

# Alternative using scikit-learn. Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values) # fit_transport, shortcut for fit and transform separately
print(y)
# Reverse mapping using LabelEncoder
y = class_le.inverse_transform(y)
print(y)

# Performing one-hot encoding on nominal features (color)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# we can't stop here and feed this to our classifiers, as it is one of the most common mistakes
# in dealing with categorical data.
# Although color values don't come in any particular order, common classification models,
# will assume that red > green > blue.

# Common workaround is one-hot encoding: create a new dummy feature for each unique value in the nominal feature column.
# i.e. convert color feature into three new features: blue, green, and red. And use binary values to indicate color of an example
# i.e. blue=1, green=0, red=0
# for this transformation we use OneHotEncoder from scikit-learn
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray() # reshape selected column.
# -1 is a placeholder, means 'unspecified'. When used on one dimesion of the reshape operation,
# NumPy automatically calculates the size of that dimension based on the length of the array
# and other specified dimension (in this case, '1').
# So reshape(-1, 1)  reshapes the array to have one column ('1') and an automatically calculated number of rows ('-1')
# converting a 1D array to a 2D column vector.
print(color_ohe)

# We used OneHotEncoder to only a single column to avoid modifying other two columns in the array
# If we want to selectively transform columns in a multi-feature array, we can use ColumnTransformer,
# which accepts a list of (names, transformer, columns(s)) tuples as follows:
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(),  [0]),
                              ('nothing', 'passthrough', [1, 2])
                              ])
c_transf = c_transf.fit_transform(X).astype(float)
#c_transf.fit_transform(X).astype(float)
print(c_transf)
# preceding code specified modifying only first column,
# and leaving the other two columns untouched 'passtrough'


# A more conveniet way to create dummy feature via one-hot encoding is to use the get_dummies method from pandas.
# When used on DataFrame, this method only converts string columns and leave all other columns unchanged.
print(pd.get_dummies(df[['price', 'color', 'size']]))

# multicollinearity guard in get_dummies
# since feature can be highly correlated, matrcies can be computationally difficult to invert.
# this can lead to numerically unstable estimates. So we reduces the correlation among variables by simply removing
# feature columns from the one-hot encoded array. i.e. removing color_blue, still preserve the feature information,
# since we still have information about color_green=0 and color_red=0, which implies the observation for color_blue
print(pd.get_dummies(df[['price', 'color', 'size']],
                     drop_first=True))

# multicollinearity guard for the OneHotEncoder. Drop redundant column
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([('oneshot', color_ohe, [0]),
                              ('nothing', 'passthrough', [1, 2])
                              ])
c_transf = c_transf.fit_transform(X).astype(float)
print(c_transf)

# ## Optional: Encoding Ordinal Features

# If we are unsure about the numerical differences between the categories of ordinal features, or the difference between two ordinal values is not defined, we can also encode them using a threshold encoding with 0/1 values. For example, we can split the feature "size" with values M, L, and XL into two new features "x > M" and "x > L". Let's consider the original DataFrame:



df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
df


# We can use the `apply` method of pandas' DataFrames to write custom lambda expressions in order to encode these variables using the value-threshold approach:



df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

del df['size']
print(df)

