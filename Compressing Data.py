import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Unsupervised dimensionality reduction via principal component analysis

# The main steps behind principal component analysis





# Extracting the principal components step-by-step
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Extracting the principal components step by step

# Step 1: Standardize the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Construct covariance matrix (symmetric d x d dimensional matrix, where d is number of dimensions in our dataset)
# Note, positive covariance between two features indicate increase or decrease together (directly proportional)
# Whereas negative covariance indicates features vary in opposite directions (inversely proportional)
# The eigenvectors of the covariance matrix represents the principal components (direction of maximum variance)
# Whereas corresponding eigenvalues define their magnitude.
# In the case of our Wine dataset, we would have 13 eigenvectors and eigenvalues from a 13x13-dimensional covariance matrix

# Step 2: Construct eigenvector
cov_mat = np.cov(X_train_std.T)

# Step 3: Obtain the eigenvalues and eigenvectors of the covariance matrix / Perform eigendecomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)

# In order to reduce dimensionality of dataset by compressing it onto a new feature subspace,
# we only select the subset of eigenvectors (principal components) that conntains most of the informationa (variance)
# Since the eigenvalues  defines the magnitude of eigenvectors, we sort them by decreasing magnitude:
# We are interested in the top k eigenvectors based on the values of their corresponding eigenvalues.

# Before doing so we plot the total and variance explained ratio of eigenvalues.
# Variance explained ratio of an eigenvalue, is simply a fraction of an eigenvalue and the total sum of eigenvalues:
# Equation pp. 144 on Raschka


# Step 4: Sorting the eigenvalues by decreasing order to rank the eigenvectors
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,14), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Although the plot of explained variance reminds of feature importance values in in Partition Dataset.py,
# we must remember that PCA is an unsupervised method, meaning the information about the class labels are ignored.
# Remember: Random forest uses class membership information to compute node impurities,
# variance measures spread of values along a feature axis.

# Feature transformation:

# last 3 steps consist of sorting the eigenpairs by descending order of eigenvalues, construct a projection of matrix
# from the selected eigenvectors, and use the projection matrix to transform the data onto the lower dimensional subspace

# Step 5: Select k eigenvectors, which correspond to the k largest eigenvalues, where k is the dimensionality
# of the new feature subspace (k <= d)

# Make a list of (eigenvalues, eigenvectors) typles
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Collect two eigenvectors corresponding to the two largest eigenvalues
# Note that two eigenvectors have been chosen for illustrative purposes since we are going to plot the data
# in a 2-dimensional scatterplot later, and as we saw in our plot earlier, the first two captured
# 60% of the variance in the Wine dataset. In practice the numebr of principal components is a tradeoff between
# computational efficieny and performance of a classifier

# Step 6: Construct a projection matrix, W, from the top k eigenvectors
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
# Preceding code created a 13x2-dimensional projection matrix W, from the top two eigenvectors.

# Step 7: Transform the d-dimensional input dataset, X, using the projection matrix, W,
# to obtain the new k-dimensional feature space

# projection matrix trnasforming the entire 123x13-dimensional training datasaet onto the two principal components
# by calculating the matrix dot product:
X_train_pca = X_train_std.dot(w) # X' = XW

# projection matrix on a single example x (representing a 13-dimensional row vector) transforming onto PCA subspace (Principal components one and two)
# consisting of two new features
X_train_std[0].dot(w) # x' = xW

# Lastly visualize the transformed Wine training dataset now stored as a 124x2-dimensional matrix in a 2-dimensional scatterplot
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# this graph shows that the data is more spread along the first principal component,
# than the second principal component (y-axis). Which is consistent with the explained variance ratio plot earlier
# the explained variance ratio showed that approx. 1st PC accounts for approx. 40% of the variance
# whereas the 2nd PC accounted for approx 20% of the variance
# and when we look at the graph here, we can easier distinguish between class 1, 2, and 3 using the 1st PC
# compared to if we looked at the 2nd PC (Class 1 and class 3 are at the same level).
# Althoguh we encoded the class label information for purpose of illustration, note that PCA is an unsupervised technique
# where we don't use any class label information






# Principal component analysis in scikit-learn
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# **NOTE**
# The following four code cells has been added in addition to the content to the book, to illustrate how to replicate
# the results from our own PCA implementation in scikit-learn:

"""
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_




plt.bar(range(1, 14), pca.explained_variance_ratio_, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)




plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
"""

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

# initializing the PCA transformer and logistic regression estimator
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1,
                        solver='lbfgs')

# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# fitting the logistic regression model on the reduced dataset:
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# for completeness sake plot decision regions of the LR on transformed test dataset
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# explained variances ratios of the PCs initialized using the n_components parameter set to None, so are PCs
# are kept and the explained variance ratio can be accessed via the explained_variance_ratio_ attribute
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
#print(pca.explained_variance_ratio_)
# note by setting n_components=None we initialize the PCA class so it will return all PCs in a sorted order, instead of
# performing a dimensionality reduction



# Assessing feature contributions

# compute 13x13-dimensional loadings matrix by multiplying eigenvectors by square root of the eigenvalues:
# here loadings means how much each original feature contributes to a given principal component
loadings = eigen_vecs * np.sqrt(eigen_vals)

# plot loadings for the first principal component, loadings[:, 0], meaning first column in our matrix:
fig, ax = plt.subplots()

ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
#plt.savefig('figures/05_05_02.png', dpi=300)
plt.show()

# we see, Alcohol has a negative correlation with the 1st PC component (ca. -0.3),
# whereas Malic acid has a positive correlation (ca. 0.54). Note the value 1 describe a perfect positive correlation,
# whereas value -1 corresponds to perfect negative correlation.

# instead of computing factor loadings using our own PCA implementation, we obtain the loadings from a fitted scikit-learn
# PCA object similarly, where pca.components_ represent the eigenvectors and pca.explained_variance_ represent eigenvalues
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# compare scikit-learn PCA loadings with the ones created before, we create a similar bar plot:
fig, ax = plt.subplots()

ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
#plt.savefig('figures/05_05_03.png', dpi=300)
plt.show()







# Supervised data compression via linear discriminant analysis

# Principal component analysis versus linear discriminant analysis
# LDA is similar to PSA, but where PCA attempts to find the orthogonal component axes of maximum variance in the dataset
# the goal with LDA is to find the feature subspace that optimizes class separability.
# Both are linear transformation techniques, but while PCA is unsupervised algorithm, LDA is supervised.
# PCA can be better for classification if each class consist of only small number of samples. Otherwise LDA is superior






# ## The inner workings of linear discriminant analysis

# Step 1: Standardize dataset (already done)

# ## Computing the scatter matrices

# Step 2: Calculate the mean vectors for each class:
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4): # here we look at classes between {1,2,3} hence the range(1,4). When then use our y_train containing the true values with the label variable to find the rows that match
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0)) # find the rows in our X_train_std where our y_train class label matches the labels between the classes in {1,4}
    print(f'MV {label}: {mean_vecs[label - 1]}\n')

# Compute the within-class scatter matrix S_w using the mean vectors:
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    #print(mv)
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1) # you take the examples/rows belong to class and minus the mean values
        print(row)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

# Better: covariance matrix since classes are not equally distributed:

# this shows our assumption that the class labels in the training dataset is uniformly distributed,
# when we compute the scatter matrices, is violated.
print('Class label distribution:', np.bincount(y_train)[1:])

# So we want to scale the individual scatter matrices S_i before we sum them up as the scatter matrix S_w
# if we divide the scatter matrices by the number of class-examples n_i, we see that the scatter matrix is actually
# the same as computing the covariance matrix - covariance matrix is a normalized version of the scatter matrix. pp. 157
# Step 3: Construct between-class scatter matrix S_B, and within-class scatter matrix S_W
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

# Compute the between-class scatter matrix:
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)  # make column vector

d = 13  # number of features
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')



# Selecting linear discriminants for the new feature subspace

# Step 4: Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# **Note**:
#
# Above, I used the [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) function to decompose the symmetric covariance matrix into its eigenvalues and eigenvectors.
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     This is not really a "mistake," but probably suboptimal. It would be better to use [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) in such cases, which has been designed for [Hermetian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix). The latter always returns real  eigenvalues; whereas the numerically less stable `np.linalg.eig` can decompose nonsymmetric square matrices, you may find that it returns complex eigenvalues in certain cases. (S.R.)
#

# Sort eigenvectors in descending order of the eigenvalues:



# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Step 5: Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, align='center', label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
plt.ylabel('"Discriminability ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# the f irst two linear discriminants alone capture 100% of the useful information in the Wine training dataset

# Step 6: Choose k eigenvectors that correspond to the k largest eigenvalues.
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


# Step 7: Projecting examples onto the new feature space
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1] * (-1),
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



# ## LDA via scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# Using logistic regression model with LDA we misclassified one example from class 2

# By lowering the regularization strength, we can shift the decision boundaries so the Logistic Regression model
# classifies all examples in the training dataste correctly.
# But more importantly lets see the results for the test dataset

X_test_ld = lda.transform(X_test_std)
plot_decision_regions(X_test_ld, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# hence, the Logistic Regression classifier is able to get a perfect accuracy score for classfying examples in the
# test dataset by only using a two-dimensional feature subspace instead of the original 13 Wine features.





# # Nonlinear dimensionality reduction techniques


# ### Visualizing data via t-distributed stochastic neighbor embedding
# in a nutshell t-SNE models data points based on their pair-wise distances in the high-dimensional (original) feature space.
# Then finds a probability distribtuion of pair-wise distances in the new lower-dimensional space that is close
# to the probability distribution of pair-wise distances in the original space

# the idea with t-SNE is to embed data points into a lower-dimensional spaces
# such that the pairwise distances in the original space are preserved
from sklearn.datasets import load_digits
digits = load_digits() # load low-resolution handwritten digits (numbers 0-9)

# digits are 8x8 grayscale images. Following code plots the first four images in the dataset,
# which consist of 1797 images in total
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

# digits.data attribute lets us acccess a tabular version of this dataset, where examples are represented as rows, and columns as pixels
print(digits.data.shape)

# Assign features (pixels) to a new variable X_digits and the labels to another new variabl y_digits:
y_digits = digits.target
X_digits = digits.data

# Import t-SNE class from scikit-learn and fit a new tsne object. Use fit_transform to perform t-SNE fit and data transformation in one step:
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)
# this projected the 64-dimensional dataset onto a 2-dimensional spaces. init='pca' initializes the t-SNE embedding
# using PCA as it is recommended. Note t-SNE includes additional hyperparameters such as perplexity and learning rate
# which we omitted in the example (we use scikit-learn default values), but in practice these parameters have to be researched

# Visualize the 2D t-SNE:
import matplotlib.patheffects as PathEffects
def plot_projection(x, colors):

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])

    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),
                              PathEffects.Normal()])
plot_projection(X_digits_tsne, y_digits)
plt.show()
# Like PCA, t-SNE is an unsupervised method, and the preceding code we use class labels y_digits (0-9) only for visualization
# via the functions color argument. Matplotlib's PathEffects are used for visual purposes, such that the class label is displayed
# in the center (via np.median) of the data points belonging to each respective digit
