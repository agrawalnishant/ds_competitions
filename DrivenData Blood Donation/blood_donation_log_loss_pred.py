from __future__ import division
# coding: utf-8

# In[47]:

# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from time import time

from sklearn import cross_validation

from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[48]:

donation_data_train = pd.read_csv("bd_train.csv")
donation_data_test = pd.read_csv("bd_test.csv")
print "Student data read successfully!"
print "Training Data:\n", donation_data_train.head()
print "\nTest Data:\n", donation_data_test.head()


# In[49]:

n_donors = donation_data_train.shape[0]
n_features = donation_data_train.shape[1]-2
print "Total number of donors: {}".format(n_donors)
print "Total number of features: {}".format(n_features)


# In[50]:

# Extract feature (X) and target (y) columns
feature_cols = list(donation_data_train.columns[1:-1])  # all columns but last are features
target_col = donation_data_train.columns[-1]  # last column is the target/label
print "Target column: {}".format(target_col)
print "Feature column(s):-\n{}".format(feature_cols)

axes = plt.gca()


X_all = donation_data_train[feature_cols]  # feature values for all students
y_all = donation_data_train[target_col]  # corresponding targets/labels
X_all_test=donation_data_test[feature_cols] 


plt.scatter(X_all['Months since Last Donation'],X_all['Months since First Donation'])
plt.ylabel("Months since Last Donation")
plt.xlabel("Months since First Donation")
plt.show()


# In[51]:


from sklearn.decomposition import RandomizedPCA

# Reduce features
pca = RandomizedPCA(whiten=True).fit(X_all)
print "Visualizing all features"
pc_df=pd.DataFrame({"pca":pca.explained_variance_ratio_})
plt.plot(pc_df)
plt.ylabel('pca (exp vari. ratios)')
plt.xlabel('features')
plt.show()

print pca.components_

print "Since variance becomes negligible after 2, hence will use n_components=2 for pca"

# Reduce features
pca = RandomizedPCA(n_components=2,whiten=True).fit(X_all)

t0 = time()
X_train_pca = pca.transform(X_all)
X_test_pca = pca.transform(X_all_test)
print "done in %0.5fs" % (time() - t0)
print "X_test_pca: ", X_test_pca.shape


# In[52]:

from sklearn import grid_search
from sklearn.metrics import log_loss


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

param_dist = {"n_estimators": [50,80,100,120],
              "learning_rate": [0.8,0.9,1.0],
              "algorithm": ["SAMME", "SAMME.R"]}

# build a classifier
dt_clf=DecisionTreeClassifier( min_samples_leaf=1,max_depth=1,max_features=2)
clf = AdaBoostClassifier(dt_clf)

# run grid search
gs_clf = GridSearchCV(clf, param_grid=param_dist,cv=4)
gs_clf.fit(X_train_pca,y_all)

print "best: ", gs_clf.best_estimator_
y_preds = gs_clf.predict_proba(X_test_pca)
print "sig f1", f1_score(y_all,gs_clf.predict(X_train_pca),pos_label=1) 

print X_all_test.iloc[:,0].shape
print y_preds.shape
print "    "

#print y_preds


# In[53]:

for i in range(0,len(X_all_test)):
    print donation_data_test.iloc[i,0],",","{:.5f}".format(y_preds[i,1])


# In[ ]:



