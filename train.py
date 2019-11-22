import pickle
import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt
#import os
#import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from math import sqrt
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import r2_score
from IPython.display import display
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn import datasets
#from sklearn.decomposition import PCA
#from sklearn.datasets import load_digits
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# roc curve and auc score
#from sklearn.datasets import make_classification
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import glob

#import seaborn as sns
from plot_utils import plot_confusion_matrix, plot_roc_curve

def train(training1="training_all.csv"):
    facedata = pd.read_csv(training1, index_col=0)
    lab = facedata.label

    features = facedata.drop('label', axis = 1)
    #The below command prints a table with statistics for each numerical column in our dataset
    features.describe().to_csv("description.csv")
    """
    Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s 
    distribution, excluding NaN values.

    Analyzes both numeric and object series, 
    as well as DataFrame column sets of mixed data types. The output will vary depending on what is provided. Refer to the notes below for more detail.
    """
    X = pd.DataFrame(features)
    ##print (X.iloc[0])
    Y = pd.Series(lab)
    X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size = 0.3)

    ##print (X_train.shape)

    y_test.replace(0,np.nan)
    y_test.replace(1,np.nan)
    y_test.replace(y_test.values,np.nan)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    ##print (X_train.shape)
    mlp.fit(X_train, y_train.values.ravel())

    filename = 'finalized_model.txt'
    pickle.dump(mlp, open(filename, 'wb'))
    predictions = mlp.predict(X_test)
    ####print("confusion matrix (test prediction) =")

    mat=confusion_matrix(y_test,predictions)
    #print(mat)

    ##print("classification report (y test pred iction")
    classifi_report=classification_report(y_test,predictions)
    ##print(classifi_report)

    auc = roc_auc_score(y_test, predictions)
    ##print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    #print((y_test,predictions))
    plot_roc_curve(fpr, tpr)
    #plt.show()
    plot_confusion_matrix(mat, [0,1],["non-face","face"])
    #plt.show()

def decision(test, filename = 'finalized_model.txt'):
    ##print (test)
    facedata = test
    # print(facedata)
    lab = facedata.label
    features = facedata.drop('label', axis=1)
    # The below command prints a table with statistics for each numerical column in our dataset
    display(features.describe())
    # print(type(features.describe()))
    #features.describe().to_csv("description.csv")
    x_test = pd.DataFrame(features)
    y_test = pd.Series(lab)
    #print("test data=")
    #print (x_test.shape, y_test.shape)
    y_test.replace(y_test.values, np.nan)
    loaded_model = pickle.load(open(filename, 'rb'))

    #print (x_test.shape, y_test.shape)
    result = loaded_model.score(x_test, y_test)
    y_pred = loaded_model.predict(x_test)
    #print(result)
    #print(y_pred)
    return result, y_pred

if __name__ == "__main__":
    train()