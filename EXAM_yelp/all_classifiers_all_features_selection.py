
import pandas as pd
import numpy as np
import os
import time
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectFromModel


###--------------------------------------###
###------------DATA PREP-----------------###
###--------------------------------------###
warnings.filterwarnings("ignore", category=FutureWarning) # Remove warnings 
pd.set_option('mode.chained_assignment', None) # Remove warnings 

# Setting working directory 
#os.chdir("/Users/anitakurm/Downloads/yelp_dataset") # Anita
os.chdir("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data") # Maria

#read data
df = pd.read_excel('yelp_subset_features.xlsx', sep=';', encoding='latin-1', nrows=10000)

# Preparing data
data = df.dropna()
data = data.drop('text', axis =1)

y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values


# Calulating correlations 
cor = data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
matrix = np.triu(data.corr())
ax = sns.heatmap(data.corr(), square=True, mask = matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('All features')




###--------------------------------------###
###------------CROSS VALIDATION ---------###
###--------------------------------------###
# create classifiers
names = ["3-Nearest Neighbors", 
        "5-Nearest Neighbors",
        "7-Nearest Neighbors",
        "Logistic Regression",
        "Logistic Regression weighted classes",
        "SVM (gamma=auto)",
        "Decision Tree (max_depth = None)", 
        "Random Forest (max_depth = None, trees = 100)", 
        "Random Forest (max_depth = 2)",
        "Naive Bayes (priors = [0.75, 0.25])",
        "Naive Bayes no priors",
        "LDA"]
         
classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    LogisticRegression(),
    LogisticRegression(class_weight='balanced'),
    SVC(gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    RandomForestClassifier(max_depth=2, random_state=0),
    GaussianNB(priors = [0.75, 0.25]),
    GaussianNB(),
    LDA()
    ]

# defines scoring measures
scoring = {'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro', 
           'balanced_accuracy': 'balanced_accuracy', 
           'f1_weighted': 'f1_weighted'
           }


results_all = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_all.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_all
all_results = pd.DataFrame(results_all, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
all_results.to_csv('classifier_performances.csv')





###--------------------------------------###
###------------FEATURE SELECTION---------###
###--------------------------------------###


#---------from Maria's script: method = LOGISTIC REGRESSION----------#

#Running the feature selection on the chosen method """

""" TO DO: CHOOSE ONE OF THE BELOW METHODS FOR MODEL SELECTION """
chosen_method = LogisticRegression

""" Running the feature selection on the chosen method """
y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values

method = chosen_method(C=0.01, penalty="l1", dual=False).fit(X, y)
model=SelectFromModel(method, prefit=True)
list_true = model.get_support()
X_new = model.transform(X)

# Returns a dataframe with chosen features     
df_features = data.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns, columns=['chosen'])
chosen_features = feat_bool[feat_bool['chosen']==True] 
needed_data = data[[i for i in chosen_features.index]]
needed_data['useful_dummy'] = data['useful_dummy']

# Calulating correlations 
cor = needed_data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
correlations = cor_target[cor_target>0.25]
print(correlations)


#CORRELATION MATRIX
matrix = np.triu(needed_data.corr())
ax = sns.heatmap(needed_data.corr(), mask=matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('Features selected via Logistic Regression')


### RUNNING CROSS validation on logistic regression features (needed data)
y = needed_data['useful_dummy'].values
X = needed_data.drop('useful_dummy', axis=1).values

results_lr_features = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_lr_features.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_lr_features
lr_results = pd.DataFrame(results_lr_features, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
lr_results.to_csv('logistic_features_performances.csv')


#----------- from Maria's script: method = LINEAR SVC -------------#

chosen_method = LinearSVC

""" Running the feature selection on the chosen method """
y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values


method = chosen_method().fit(X, y)
model=SelectFromModel(method, prefit=True)
list_true = model.get_support()
X_new = model.transform(X)

# Returns a dataframe with chosen features     
df_features = data.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns, columns=['chosen'])
chosen_features = feat_bool[feat_bool['chosen']==True] 
needed_data = data[[i for i in chosen_features.index]]
needed_data['useful_dummy'] = data['useful_dummy']

# Calulating correlations 
cor = needed_data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
correlations = cor_target[cor_target>0.25]
print(correlations)


#CORRELATION MATRIX
matrix = np.triu(needed_data.corr())
ax = sns.heatmap(needed_data.corr(), mask=matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('Features selected via Linear SVC')


### RUNNING CROSS validation on Linear SVC features (needed data)
y = needed_data['useful_dummy'].values
X = needed_data.drop('useful_dummy', axis=1).values

results_lsvc_features = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_lsvc_features.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_lsvc_features
lsvc_results = pd.DataFrame(results_lsvc_features, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
lsvc_results.to_csv('linear_svc_features_performances.csv')




#----------- new: method = LASSO CV -------------#

y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values
from sklearn.linear_model import LassoCV
clf = LassoCV()
# Set a minimum threshold of 0.25
clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
list_true = model.get_support()

# Returns a dataframe with chosen features     
df_features = data.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns, columns=['chosen'])
chosen_features = feat_bool[feat_bool['chosen']==True] 
needed_data = data[[i for i in chosen_features.index]]
needed_data['useful_dummy'] = data['useful_dummy']

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# Calulating correlations 
cor = needed_data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
matrix = np.triu(needed_data.corr())
ax = sns.heatmap(needed_data.corr(), annot = True, square=True, mask = matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('Features selected via Lasso')


### RUNNING CROSS validation on Lasso features (needed data)
y = needed_data['useful_dummy'].values
X = needed_data.drop('useful_dummy', axis=1).values

results_lasso_features = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_lasso_features.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_lasso_features
lasso_results = pd.DataFrame(results_lasso_features, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
lasso_results.to_csv('lasso_features_performances.csv')
