#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:06:54 2022

@author: kerry
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as m
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB


#show all the columns in data set when printing
pd.set_option('display.max_columns', None)

sep = "\n----------------------------------------------------\n"

df = pd.read_csv('full_dataset.csv')
print(df.head())

#replace categorical variables with numeric
genres = ['classical', 'country', 'jazz', 'pop', 'rap', 'rock', 'techno', 'world']
genres_num = [1,2,3,4,5,6,7,8]
df['genre'].replace(genres, genres_num, inplace=True)
df['explicit']. replace([True,False], [1,0], inplace=True)


#replace popularity with binary categorical variable
mean_pop = df['popularity'].mean()
median_pop = df['popularity'].median()
std_pop = df['popularity'].std()


print(df['popularity'].describe())
og_pop = df['popularity'].copy()


df.loc[df['popularity'] < mean_pop + std_pop,'popularity'] = 0
df.loc[df['popularity'] >= mean_pop + std_pop, 'popularity'] = 1


df.drop(['is_local', 'name'], axis=1, inplace=True)

#stratifiedSample = df.groupby('genre', group_keys=False).apply(lambda x: x.sample(frac=0.1))
#df = stratifiedSample


pMatrix = df.corr()
print(pMatrix)


#df.drop(['time_signature','mode'], axis=1, inplace=True)
#least correlated with popularity -- 'key' , 'mode', 'liveness','valence', 'genre', 'tempo'
#most correlated - energy, loudness acousticness



RFmodel = RandomForestClassifier()
RFmodel_acc = []

train, test = train_test_split(df,test_size=0.2, random_state=0)
# get X and y
XTrain = train.drop(["popularity"],axis=1)
XTest = test.drop(["popularity"], axis=1)
yTrain = train["popularity"]
yTest = test["popularity"]

feature_list = list(XTrain.columns)


#Random Forest
RFmodel.fit(XTrain, yTrain)
yPred = RFmodel.predict(XTest)
yPred_train = RFmodel.predict(XTrain)
RFmodel_acc.append(accuracy_score(yTest, yPred))


print(f"{sep}Accuracies for Random Forest Test: ", accuracy_score(yTest, yPred))
print("Average accuracy for Random Forest Training ", accuracy_score(yTrain, yPred_train), "\n")

print("Training confusion matrix\n", confusion_matrix(yTrain, yPred_train))
print("Testing confusion matrix:\n", confusion_matrix(yTest, yPred))

print("Training classification report:\n", classification_report(yTrain, yPred_train))
print("Test classification report:\n", classification_report(yTest, yPred))
train_df_model = train.copy()
train_df_model['yPred'] = yPred_train
train_df_model['og_pop'] = og_pop
train_df_model.to_csv("train_model.csv")

importances = list(RFmodel.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]



# Creates logistic and SVM models along with accuracy arrays.

BCModel = BaggingClassifier()
BCmodel_acc =  []
ABModel = AdaBoostClassifier()
ABmodel_acc = []
RFmodel = RandomForestClassifier()
RFmodel_acc = []
RFmodel2 = RandomForestClassifier(max_depth=50,n_estimators=150, max_features=4)
RFmodel2_acc = []
DTmodel = DecisionTreeClassifier()
DTmodel_acc =[]
QDmodel = QuadraticDiscriminantAnalysis()
QDmodel_acc = []
MLPmodel = MLPClassifier()
LGmodel = LogisticRegression()
LGmodel_acc = []
SVMmodel = SVC()
SVMmodel_acc = []
KNNmodel = KNeighborsClassifier()
KNNmodel_acc = []
MLPmodel_acc =[]
GPmodel = GaussianProcessClassifier()
GPmodel_acc = []


# Iterates through different splits.
for train_index, test_index in kfSplit:
    #create training and test data
    yTrain = df['popularity'].values[train_index]
    XTrain = df.drop(['popularity'], axis=1).values[train_index] # Creates training data.
    yTest = df['popularity'].values[test_index]
    XTest = df.drop(['popularity'], axis=1).values[test_index] # Creates testing data.
    
    #Bagging
    BCModel.fit(XTrain, yTrain) # Fits logistic model.
    yPred = BCModel.predict(XTest) # Predicts testing data class.
    BCmodel_acc.append(accuracy_score(yTest, yPred)) # Accuracy of logistic model on testing data.
    
    #AdaBoost
    ABModel.fit(XTrain, yTrain) # Fits SVM model.
    yPred = ABModel.predict(XTest) # Predicts class of testing data.
    ABmodel_acc.append(accuracy_score(yTest, yPred))# Accuracy of SVM model on testing data.
    
    #Random Forest
    RFmodel.fit(XTrain, yTrain)
    yPred = RFmodel.predict(XTest)
    RFmodel_acc.append(accuracy_score(yTest, yPred))
    #Random Forest
    RFmodel2.fit(XTrain, yTrain)
    yPred = RFmodel2.predict(XTest)
    RFmodel2_acc.append(accuracy_score(yTest, yPred))
    
    #Decision Tree
    DTmodel.fit(XTrain,yTrain)
    yPred = DTmodel.predict(XTest)
    DTmodel_acc.append(accuracy_score(yTest, yPred))
    
    #Quadradic Discriminant Tree
    QDmodel.fit(XTrain,yTrain)
    yPred = QDmodel.predict(XTest)
    QDmodel_acc.append(accuracy_score(yTest, yPred))
    

    #Neural Network Tree
    MLPmodel.fit(XTrain,yTrain)
    yPred = MLPmodel.predict(XTest)
    MLPmodel_acc.append(accuracy_score(yTest, yPred))
    
    #Gaussian Process Tree
    GPmodel.fit(XTrain,yTrain)
    yPred = GPmodel.predict(XTest)
    GPmodel_acc.append(accuracy_score(yTest, yPred))
    # logistic    
    LGmodel.fit(XTrain,yTrain)
    yPred = LGmodel.predict(XTest)
    LGmodel_acc.append(accuracy_score(yTest, yPred))

    #SVM
    SVMmodel.fit(XTrain,yTrain)
    yPred = SVMmodel.predict(XTest)
    SVMmodel_acc.append(accuracy_score(yTest, yPred))
    
    # K-means
    KNNmodel.fit(XTrain,yTrain)
    yPred = KNNmodel.predict(XTest)
    KNNmodel_acc.append(accuracy_score(yTest, yPred))


print("Accuracies for Bagging Classifier model using 5 Fold Validation: ", BCmodel_acc)
print("Average accuracy for bagging Classifier = ", np.average(BCmodel_acc), "\n")

print("Accuracies for AdaBoost Classifier model using 5 Fold Validation: ", ABmodel_acc)
print("Average accuracy for AdaBoost Classifier model = ", np.average(ABmodel_acc), "\n")

print("Accuracies for Random Forest Classifier model using 5 Fold Validation: ", RFmodel2_acc)
print("Average accuracy for Random Forest Classifier model = ", np.average(RFmodel2_acc), "\n")

print("Accuracies for Random Forest Classifier model using 5 Fold Validation: ", RFmodel_acc)
print("Average accuracy for Random Forest Classifier model = ", np.average(RFmodel_acc), "\n")


print("Accuracies for Logistic Regression model using 5 Fold Validation: ", LGmodel_acc)
print("Average accuracy for Logistic Classifier model = ", np.average(LGmodel_acc), "\n")

print("Accuracies for SVM model using 5 Fold Validation: ", SVMmodel_acc)
print("Average accuracy for SVM Classifier model = ", np.average(SVMmodel_acc), "\n")

print("Accuracies for K-Means Classifier model using 5 Fold Validation: ", KNNmodel_acc)
print("Average accuracy for K-means Classifier model = ", np.average(KNNmodel_acc), "\n")

print("Accuracies for Decision Tree Classifier model using 5 Fold Validation: ", DTmodel_acc)
print("Average accuracy for Decison Tree Classifier model = ", np.average(DTmodel_acc), "\n")

print("Accuracies for Neural Network Classifier model using 5 Fold Validation: ", MLPmodel_acc)
print("Average accuracy for Neural Network Classifier model = ", np.average(MLPmodel_acc), "\n")

print("Accuracies for Gaussian Process Classifier model using 5 Fold Validation: ", GPmodel_acc)
print("Average accuracy for Gaussian Process Classifier model = ", np.average(GPmodel_acc), "\n")
 
print("Accuracies for Quadradic Discriminant Classifier model using 5 Fold Validation: ", QDmodel_acc)
print("Average accuracy for Quadradic Discriminant Classifier model = ", np.average(QDmodel_acc), "\n")
"""
"""
n=5
kf = KFold(n_splits=n, shuffle=True)
kfSplit = kf.split(df)

depth_range = range(5,15,1)
est_range = range (80,90,10)
feat_range = range(8,10,1)

values = []
max_acc = 0
for d in depth_range:
    for e in est_range:
        for f in feat_range:

            RFmodel = RandomForestClassifier(max_depth=d,
                                             n_estimators=e, 
                                             max_features=f, 
                                             random_state = 0)
            RFmodel_acc = []
            
            n=5
            kf = KFold(n_splits=n, shuffle=True)
            kfSplit = kf.split(df)
            tot_acc = 0
            for train_index, test_index in kfSplit:
                yTrain = df['popularity'].values[train_index]
                XTrain = df.drop(['popularity'], axis=1).values[train_index] # Creates training data.
                yTest = df['popularity'].values[test_index]
                XTest = df.drop(['popularity'], axis=1).values[test_index] # Creates testing data.
                
                #create training and test data
                yTrain = df['popularity'].values[train_index]
                XTrain = df.drop(['popularity'], axis=1).values[train_index] # Creates training data.
                yTest = df['popularity'].values[test_index]
                XTest = df.drop(['popularity'], axis=1).values[test_index] # Creates testing data.
                
                #Random Forest
                RFmodel.fit(XTrain, yTrain)
                yPred = RFmodel.predict(XTest)
                acc = accuracy_score(yTest, yPred)
                tot_acc += acc
                RFmodel_acc.append(acc)
            ave_acc = tot_acc/n
            if (ave_acc > max_acc):
                max_acc = ave_acc
                max_d = d
                max_e = e
                max_f = f
            values.append([d,e,f,ave_acc])
            
            print(values[-1])
            print("Accuracies for Random Forest Classifier model using 5 Fold Validation: ", RFmodel_acc)
            print("Average accuracy for Random Forest Classifier model = ", np.average(RFmodel_acc), "\n") 


print(max_acc, max_d, max_e, max_f)
v = pd.DataFrame(values, columns = ['d','e','f','acc'])
print(v)
print(v['acc'].max())

