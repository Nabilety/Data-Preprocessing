import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv('wdbc.data', header=None, encoding='utf-8')
except FileNotFoundError:
    print("File not found! Please make sure 'wine.data exists")

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)
print(le.transform(['M', 'B']))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

# Combining transformers and estimators in a pipeline (wrapper) - chain model fitting and data transformation for training and test datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression())
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

# # Using k-fold cross validation to assess model performance

# ## K-fold cross-validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'Fold: {k + 1:02d}, '
          f'Class distr.: {np.bincount(y_train[train])}, '
          f'Acc.: {score:.3f}')
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')

# cross_val-score can distribute the evaluations on different folds across multiple CPUs. n_jobs = 1 uses only 1 CPU.
# Setting n_jobs=2 we can distribute 10 rounds of cross-validation to two CPU, n_jobs=-1 use all available CPUs
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')

# Debugging algorithms with learning curves

# Diagnosing bias and variance problems with learning curves
# learning curve function
from sklearn.model_selection import learning_curve
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2',
                                           max_iter=10000))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10,
                                                        n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()

# Addressing over- and underfitting with validation curves
# instead of plotting training and test accuracies as function of samples size, we vary the values of the model parameters
# for example, inverse regularization parameter, C
# validation curve function
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                             X=X_train,
                                             y=y_train,
                                             param_name='logisticregression__C',
                                             param_range=param_range,
                                             cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')
plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()
# Although the differences in the accuracy for varying values of C are subtle, we can see that the model
# slightly underfits the data when we increase the regularization strength (small values of C). However,
# for large values of C, it means lowering the strength of regularization, so the model tends to slightly
# overfit the data. In this case, the sweet spot appears to be between 0.01 and 0.1 of the C value.




# Fine-tuning machine learning models via grid search

# Tuning hyperparameters via grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
# RBF kernel SVM model with svc__C = 100.0 yields best k-fold cross-validation accuracy

clf = gs.best_estimator_
# clf.fit(X_train, y_train)
# note that we do not need to refit the classifier
# because this is done automatically via refit=True when we created the GridSearchCV class.
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')


# ## Exploring hyperparameter configurations more widely with randomized search
param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]

# instead of providing a discrete list , the power with RSCV is we can feed it a distribution to sample from instead:
# for example, using loguniform distribution instead of regular uniform distribution will make sure
# that in sufficiently large numbers of trials, the same number of samples will be drawn from the [0.0001, 0.001] range,
# as for example for the [10.0, 100.0] range.
param_range = scipy.stats.loguniform(0.0001, 1000.0)

# Check this behavior by executing following two lines drawing 10 random samples from this distribution via the rvs(10):
np.random.seed(1)
param_range.rvs(10)

from sklearn.model_selection import RandomizedSearchCV
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
""" Alternative grid and distribution define
param_grid_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear']
}    
param_dist_random = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.0001, 0.1),
    'kernel': ['rbf', 'linear']
}
"""
rs = RandomizedSearchCV(estimator=pipe_svc,
                        param_distributions=param_grid,
                        scoring='accuracy',
                        refit=True,
                        n_iter=20,
                        cv=10,
                        random_state=1,
                        n_jobs=1)

rs = rs.fit(X_train, y_train)
print(rs.best_score_)
print(rs.best_params_)

# More resource-efficient hyperparameter search with successive halving
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
hs = HalvingRandomSearchCV(pipe_svc,
                           param_distributions=param_grid,
                           n_candidates='exhaust',
                           resource='n_samples',
                           factor=1.5,
                           random_state=1,
                           n_jobs=-1)
# resource: specify the training set size to vary between the rounds.
# factor: determine how many candidates we eliminate in each round. i.e. factor=2, we eliminate half of the candidates
# factor=1.5 means only 100%/1.5 = 66% of the candidates make it into the next round.
# n_candidates: instead of choosing a fixed number of iterations as in RandomizedSearchCV, we set n_candidates='exhaust'
# which will sample the numer of hyperparameter configurations such that the maximum number of resources (here: training examples)
# are used in the last round.

hs = hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)

clf = hs.best_estimator_
print(f'Test accuracy: {hs.score(X_train, y_train):.3f}')


# ## Algorithm selection with nested cross-validation
param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')

# Use nested cross-validation to compare an SVM model to a simple decision tree classifier
# (for simplicity we only tune its depth parameter):
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')


# # Looking at different performance evaluation metrics

# ...

# ## Reading a confusion matrix
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
ax.xaxis.set_ticks_position('bottom')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

# ### Additional Note

# Remember that we previously encoded the class labels so that *malignant* examples are the "postive" class (1),
# and *benign* examples are the "negative" class (0):

# ## Optimizing the precision and recall of a classification model
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = recall_score(y_true=y_test, y_pred=y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = f1_score(y_true=y_test, y_pred=y_pred)
print(f'F1: {f1_val:.3f}')

mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f'MCC: {mcc_val:.3f}')


from sklearn.metrics import make_scorer
# remember positive class in scikit-learn is the class labeled as class 1.
# if we wanna specify a different positive label, we can construct our own scorer via make_scorer function
# which we then can provide as argument to the scoring parameter in GridSearchCV (in this exampling using f1_score metric)
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
scorer = make_scorer(f1_score, pos_label=0) # we set positive class label as class 0
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# ## Plotting a receiver operating characteristic
from sklearn.metrics import roc_curve, auc
from numpy import interp

# note althought we are going to use the logistic regression pipeline, we only are using two features this time.
# this will make the classification task more challenging for the classifier, by withholding useful information contained in other features
# so that the resutling ROC curve becomes more interesting. For that reduce we are also reducing number of folds in StratifiedKFold validator to three
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           solver='lbfgs',
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label=f'ROC fold {i + 1} (area = {roc_auc:.2f})')

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing (area = 0.5)')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance (area = 1.0)')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ## The scoring metrics for multiclass classification
# Specifying the averaging method via the average parameter inside scoring function
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')

# ## Dealing with class imbalance
# creating imbalanced dataset of original consisting of 357 benign tumros (class 0)
# and 212 malignant tumors (class 1)
# take all 357 benign tumor examples and stack them with the first 40 malignant examples
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
# if we computed the accuracy of a model that always predict majority class (benign class 0), we would achieve 90% accuracy

y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

# in other words, when we fit a classifier on such dataset, it make sense to focus on other metrics
# than accuracy when comparing models, such as precision, recall, ROC curve, MCC, etc.

# for instance, if our priority might be to identify the majority of patients with malignant cancer
# to recommend additional screening, thus recall should be our metric of choice TP/TP+FN

# in spam filtering, where we don't want to label emails as spam if the system is not very certain,
# precision will be more appropriate metric


# Another important note, when evaluating machine learning models, class imbalance influences a learning algorithm
# during model fitting itself. Since machine learning algorithm typically optimize a reward/loss function
# that is computed as a sum over the training examples that it sees during fitting, the decision rule is likely
# going to be biased toward the majority class. In other words, the algorithm implicitly learns a model that optimizes
# the predictions based on the most abundant class in the dataset to minimize the loss or maximize the rewards during training

# one way to deal with this issue of imbalanced class proportions during model fitting is to assign a larger penalty
# to wrong predictions on the minority class. Via scikit-learn, adjusting such a penalty is as convenient as setting the
# class_weight paramter to class_weight='balanced', implemented in most classifiers.

# Alternative strategies involve upsampling the minority class, downsampling the majority class, and generation of synthetic
# training examples. There's no universally best solution as it can vary for different problem domains.
# so in practice you have to try different strategies for a given problem, evaluate the results, and choose the most appropriate technique

# Scikit learn library implements a simple resample function that can help with upsampling of the minority class
# by drawing new samples from the dataset WITH replacement.
# Following code will take minority class from our imbalanced dataset (here class 1) and repeatedly draw new samples
# From it until it contains the same number of examples as class label 0:
from sklearn.utils import resample
print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
print('Number of class 1 examples after:', X_upsampled.shape[0])

# After resampling, we stack the original class 0 samples with the upsampled class 1 subset to obtain a balanced dataset
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

# a majority vote prediction rule would consequently achieve 50 percent accuracy
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100

# Similarly we could downsample the majority class by removing training examples from the dataset.
# to perform downsampling using resample function, we can simply swap the class 1 label with class 0 in previous code


# Another technique for dealing with class imbalance is generating synthetic training examples,
# here the most widely used algorithm is Synthetic Minority Over-sampling Technique (SMOTE)


