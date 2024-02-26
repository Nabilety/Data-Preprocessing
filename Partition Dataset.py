import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

try:
    df_wine = pd.read_csv('wine.data', header=None, encoding='utf-8')
except FileNotFoundError:
    print("File not found! Please make sure 'wine.data exists")

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#randomly split X and y into separate train and test sets, with 30% to X_test and y_test, 70% to X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

# Feature scaling - bringing features onto the same scale.

# Normalization, commonly useful for bounded interval values
# Consist of rescaling feature to range between [0, 1]
mms = MinMaxScaler() # x_norm = (xi - x_min) / (x_max - x_min), here xi is particular example, x_max/min smallest/largest values in feature column
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardization most practical for ML algorithms due to zero mean and unit variance.
# Meaning, feature columns centered at mean 0, with standard deviation 1.
# x_std = (xi - μ_x) / σ_x, here μ_x is sample mean of particular column, σ_x is standard deviation of that column
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Simple example of difference between standardization and normalization
ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean()) / ex.std())
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

# note, we fit the StandardScaler class only once - on the training data, and later use those parameters to transform the test dataset and new data

# RobustScaler is another feature scaling method, but usually recommended for smaller datasets that contains many outliers


# Selecting meaningful features
# Ways to mitigate overfitting (model performing better on training dataset than test dataset):
# - Collect more training data
# - Introduce a penalty for complexity via regularization
# - Choose a simpler model with fewer parameters
# - Reduce the dimensionality of the data

# note we use liblinear optimization algorithm, since 'lbfgs' is not currently supported for L1-regularization.
lr = LogisticRegression(penalty='l1',
                        C=1.0,
                        solver='liblinear',
                        multi_class='ovr')

lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
# Since we fit the LR object on multiclass dataset with one-versus-rest (OvR).
# the first intercept belongs to the model that fits class 1 versus classes 2 and 3,
# the second is the intercept of the model that fits class 2 versus classes 1 and 3
# the third is the intercept of the model that fits class 3 versus classes 1 and 2 the int the print below
print(lr.intercept_)

# The weight array that we accessed via the lr.coef_ attribute contains three rows of weight coefficients,
# one weight vector for each class. Each row consists of 13 weights, where each weight is multiplied by
# the respective feature in the 13-dimensional Wine dataset to calculate the net input:
print(lr.coef_)

# the scikit-learn intercept_ corresponds to the bias unit,
# and the coef_ corresponds to the weight values w_j

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10. ** c, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10 ** (-5), 10 ** 5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()


# Sequential feature selection algorithms
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier

class SBS:
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1] # dimension of the features (cardinality)
        print("#############")
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score] # initial score with all features
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                print(scores)
                print(subsets)

            best = np.argmax(scores) # find position of the best scores in each score list
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        # Note we're not calculating the criterion, but simply removing the feature that is not contained in the best performing feature subset
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

# collect column indices of the three-feature subsets from the 11th position in the sbs.subsets_ attribute
# return the corresponding feature names from the column index of the Wine dataframe
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# Evaluate performance on KKN using original test dataset

# Complete feature set
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

# three-feature subset
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))

# in comparison, using three features declines the prediction accuracy slightly on our test dataset
# this can indicate those three feature do not provide less discriminatory information than the original dataset
# however, keep in mind the Wine dataset is a small dataset and prone to randomness- meaning, we split the dataset
# into training and test subsets, and how we split the training dataset further into training and validation subset.
# While reducing the number of feature didn't not increase performance,
# it shrank the dataset which can be useful in real-world applications that involve expensive computation.
# similarly we introduced a simpler model, to interpet



# Assessing feature importance with Random Forests
# Train forest of 500 trees on Wine dataset and rank the 13 features by respective importance measures.
# NB: We don't need standardized/normalized feature in tree-based models:
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
# Note feature importance values are normalized, that is they sum up to 1.0
# here the Proline, Flavanoids, Color Intensity, OD280/OD315 diffraction and alcohol concentration of wine
# are the most discriminative features in our dataset based on the average impurity decrease in the 500 decision trees.
# Interestingly two of the top-ranked feature we also include in our feature selection algorithm (alcohol concentration and OD280/OD315)

# There's an importance thing to note here though, which is that if two or more feature are highly correlated,
# one feature may be ranked very highly while the information on the other feature(s) may not be fullut captured.
# This is not a concern if we're merely interested in predictive performance, rathter than the interpretion of feature importance values


# SelectFromModel selects featured based on user-specified threshold  after model fitting, which is useful if we want to
# use RandomForest as feature selector and intermediate step in our Pipeline object. This allows us to connect different
# preprocessing steps with an estimator.

# I.e. set threshold to 0.1 to reduce dataset to the five most important features:
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of feature that meet this threshold', 'criterion:', X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

