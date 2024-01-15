import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('full_dataset.csv')
df = df.loc[:, ~df.columns.isin(['name', 'is_local'])]

le = LabelEncoder()
g = df['genre']
g = le.fit_transform(g)
df = df.loc[:, ~df.columns.isin(['genre'])]
df['genre'] = g

# pearsonMatrix = df.corr()

# Uncomment the following 2 lines to run the algorithms with stratified sample.
stratifiedSample = df.groupby('genre', group_keys=False).apply(lambda x: x.sample(frac=(1/10)))
df = stratifiedSample

# This block of code makes scatter plots.
color = ['blue', 'red', 'yellow', 'black', 'green', 'orange', 'purple', 'gray']
for col in df.columns:
    for i in range(0, 8):
        plt.scatter(df.loc[df['genre'] == i][col], df.loc[df['genre'] == i]['popularity'],
                    s=10, color=color[i], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('popularity')
    plt.legend(['Classical', 'Country', 'Jazz', 'Pop', 'Rap', 'Rock', 'Techno', 'World'])
    plt.title(col + ' vs popularity Scatter Plot')
    plt.show()

# This code block converts popularity into a binary class.
df = df.astype({"explicit": int})
avg_pop = df['popularity'].mean() + df['popularity'].std()
unpopular = df.loc[df['popularity'] < avg_pop]['popularity']
popular = df.loc[df['popularity'] >= avg_pop]['popularity']
df['popularity'] = df['popularity'].replace(unpopular.to_numpy(), 0)
df['popularity'] = df['popularity'].replace(popular.to_numpy(), 1)
df = df.astype({"popularity": int})

'''# This code block runs KNN models for different k's.
train, test = train_test_split(df, test_size=0.2, random_state=1) # Splits the data as specified.

# Creates array for k values.
k = np.array([1, 3, 5, 7, 9, 11])
knnAcc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf, m.inf]) # Array to hold accuracies of every k value.

yFit = train['popularity']
XFit = train.drop(labels='popularity', axis=1)
yTest = test['popularity']
XTest = test.drop(labels='popularity', axis=1)

# Trains KNN models for every k.
for i in range(0, len(k)):
    kModel = KNeighborsClassifier(n_neighbors=k[i], p=2, metric='minkowski') # Uses Euclidean distance.
    kModel.fit(XFit, yFit)
    yPred = kModel.predict(XTest) # Predicts the test data.
    knnAcc[i] = accuracy_score(yTest, yPred) # Accuracy of the model on the testing data.

# Plots accuracy of testing data predictions.
plt.plot(k, knnAcc)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy of KNN Using Euclidean Distance')

# Trains KNN models for every k.
for i in range(0, len(k)):
    kModel = KNeighborsClassifier(n_neighbors=k[i], p=2, metric='minkowski') # Uses Euclidean distance.
    kModel.fit(XFit, yFit)
    yPred = kModel.predict(XFit) # Predicts the training data.
    knnAcc[i] = accuracy_score(yFit, yPred) # Accuracy of the model on the training data.

# Plots the accuracy of the training data.
plt.plot(k, knnAcc)
plt.legend(['Testing Set', 'Training Set'])
plt.show()

# Array to store accuracies for KNNeighborClassifier.
knnAcc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf, m.inf])

# Trains KNN models for every k.
for i in range(0, len(k)):
    kModel = KNeighborsClassifier(n_neighbors=k[i], p=1, metric='minkowski') # Uses Manhattan distance.
    kModel.fit(XFit, yFit)
    yPred = kModel.predict(XTest) # Predicts class of testing data.
    knnAcc[i] = accuracy_score(yTest, yPred) # Accuracy of model on testing data.

# Plots the accuracy of the testing data.
plt.plot(k, knnAcc)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy of KNN Using Manhattan Distance')
kModel = 0

# Trains KNN models for every k.
for i in range(0, len(k)):
    kModel = KNeighborsClassifier(n_neighbors=k[i], p=1, metric='minkowski') # Uses Manhattan distance.
    kModel.fit(XFit, yFit)
    yPred = kModel.predict(XFit) # Predicts class of training data.
    knnAcc[i] = accuracy_score(yFit, yPred) # Accuracy of model on training data.

# Plots accuracy of KNN model on training data.
plt.plot(k, knnAcc)
plt.legend(['Testing Set', 'Training Set'])
plt.show()

# This block of code runs Logistic Regression and SVM models.
# Creates 5-Fold splits.
kf = KFold(n_splits=5, shuffle=True)
kfSplit = kf.split(df)

# Creates logistic and SVM models along with accuracy arrays.
LRModel = LogisticRegression(random_state=0, max_iter=7000)
model1_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
model2_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])

i = 0
# Iterates through different splits.
for train_index, test_index in kfSplit:
    yTrain = df['popularity'].values[train_index]
    XTrain = df.drop(labels='popularity', axis=1).values[train_index] # Creates training data.
    yTest = df['popularity'].values[test_index]
    XTest = df.drop(labels='popularity', axis=1).values[test_index] # Creates testing data.
    LRModel.fit(XTrain, yTrain) # Fits logistic model.
    yPred = LRModel.predict(XTest) # Predicts testing data class.
    model1_acc[i] = accuracy_score(yTest, yPred) # Accuracy of logistic model on testing data.
    svm.fit(XTrain, yTrain) # Fits SVM model.
    yPred = svm.predict(XTest) # Predicts class of testing data.
    model2_acc[i] = accuracy_score(yTest, yPred) # Accuracy of SVM model on testing data.
    i += 1

print("Accuracies for Logistic Regression Model using 5 Fold Validation: ", model1_acc)
print("Average accuracy for Logistic Regression Model = ", np.average(model1_acc), "\n")
print("Accuracies for Support Vector Machine (RBF) using 5 Fold Validation: ", model2_acc)
print("Average accuracy for SVM Model = ", np.average(model2_acc), "\n")

# This block of code runs a Naive Bayes model.
# Creates 5-Fold splits.
kf = KFold(n_splits=5, shuffle=True)
kfSplit = kf.split(df)

# Creates Naive Bayes Model.
NBModel = GaussianNB()
model1_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf]) # Accuracy array.

i = 0
# Iterates through different splits.
for train_index, test_index in kfSplit:
    yTrain = df['popularity'].values[train_index]
    XTrain = df.drop(labels='popularity', axis=1).values[train_index] # Creates training data.
    yTest = df['popularity'].values[test_index]
    XTest = df.drop(labels='popularity', axis=1).values[test_index] # Creates testing data.
    NBModel.fit(XTrain, yTrain) # Fits naive bayes model.
    yPred = NBModel.predict(XTest) # Predicts testing data class.
    model1_acc[i] = accuracy_score(yTest, yPred) # Accuracy of naive bayes model on testing data.
    i += 1

print("Accuracies for Naive Bayes model using 5 Fold Validation: ", model1_acc)
print("Average accuracy for Naive Bayes model = ", np.average(model1_acc), "\n")

# This block of code runs Ensemble models.
# Creates 5-Fold splits.
kf = KFold(n_splits=5, shuffle=True)
kfSplit = kf.split(df)

# Creates Bagging, AdaBoost, and Random Forest classifiers along with accuracy arrays.
BCModel = BaggingClassifier(n_estimators=100, bootstrap=True, oob_score=True)
model1_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model1_recall = 0.0
model1_f1 = 0.0
ABModel = AdaBoostClassifier(n_estimators=100)
model2_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model2_recall = 0.0
model2_f1 = 0.0
RFModel = RandomForestClassifier(criterion='gini', n_estimators=100, bootstrap=True, oob_score=True, warm_start=False)
model3_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model3_recall = 0.0
model3_f1 = 0.0
# Variables to hold test and train reports for classifiers.
test1_report = 0
train1_report = 0
test2_report = 0
train2_report = 0
test3_report = 0
train3_report = 0

i = 0
# Iterates through different splits.
for train_index, test_index in kfSplit:
    yTrain = df['popularity'].values[train_index]
    XTrain = df.drop(labels='popularity', axis=1).values[train_index] # Creates training data.
    yTest = df['popularity'].values[test_index]
    XTest = df.drop(labels='popularity', axis=1).values[test_index] # Creates testing data.
    BCModel.fit(XTrain, yTrain) # Fits Bagging model.
    yPred = BCModel.predict(XTrain)
    train1_report = classification_report(yTrain, yPred)
    yPred = BCModel.predict(XTest) # Predicts testing data class.
    test1_report = classification_report(yTest, yPred)
    model1_acc[i] = accuracy_score(yTest, yPred) # Accuracy of bagging model on testing data.
    model1_recall += recall_score(yTest, yPred) # recall of bagging model on testing data.
    model1_f1 += f1_score(yTest, yPred) # F1 of bagging model on testing data.
    ABModel.fit(XTrain, yTrain) # Fits AdaBoost model.
    yPred = ABModel.predict(XTrain)
    train2_report = classification_report(yTrain, yPred)
    yPred = ABModel.predict(XTest) # Predicts class of testing data.
    test2_report = classification_report(yTest, yPred)
    model2_acc[i] = accuracy_score(yTest, yPred) # Accuracy of AdaBoost model on testing data.
    model2_recall += recall_score(yTest, yPred)
    model2_f1 += f1_score(yTest, yPred)
    RFModel.fit(XTrain, yTrain) # Creates Random Forest Model.
    yPred = RFModel.predict(XTrain)
    train3_report = classification_report(yTrain, yPred)
    yPred = RFModel.predict(XTest)
    test3_report = classification_report(yTest, yPred)
    model3_acc[i] = accuracy_score(yTest, yPred) # Statistics of Random Forest classifier.
    model3_recall += recall_score(yTest, yPred)
    model3_f1 += f1_score(yTest, yPred)
    i += 1

# Prints all statistics of all 3 ensemble classifiers.
print("Accuracies for Bagging Classifier model using 5 Fold Validation: ", model1_acc)
print("Average accuracy for bagging Classifier = ", np.average(model1_acc))
print("Average recall = ", model1_recall / 5.0, "Average f1 = ", model1_f1 / 5.0)
print("Classification report: \n", test1_report)
print(train1_report)
print("Standard deviation for bagging classifier = ", model1_acc.std(), "\n")
print("Accuracies for AdaBoost Classifier model using 5 Fold Validation: ", model2_acc)
print("Average accuracy for AdaBoost Classifier model = ", np.average(model2_acc))
print("Average recall = ", model2_recall / 5.0, "Average f1 = ", model2_f1 / 5.0)
print("Classification report: \n", test2_report)
print(train2_report)
print("Standard deviation for AdaBoost Classifier model = ", model2_acc.std(), "\n")
print("Accuracies for Random Forest Classifier model using 5 Fold Validation: ", model3_acc)
print("Average accuracy for Random Forest Classifier model = ", np.average(model3_acc))
print("Average recall = ", model3_recall / 5.0, "Average f1 = ", model3_f1 / 5.0)
print("Classification report: \n", test3_report)
print(train3_report)
print("Standard deviation for Random Forest Classifier model = ", model3_acc.std(), "\n")

# The following code will run the bagging and random forest classifiers using warm start.
kf = KFold(n_splits=5, shuffle=True)
kfSplit = kf.split(df)

# Creates the models.
BCModel = BaggingClassifier(n_estimators=100, warm_start=True)
BC_accuracy = 0
RFModel = RandomForestClassifier(criterion='gini', n_estimators=100, warm_start=True)
RF_accuracy = 0

i = 0
for train_index, test_index in kfSplit:

    # Runs prediction on final split of the data.
    if i == 4:
        yTrain = df['popularity'].values[train_index]
        XTrain = df.drop(labels='popularity', axis=1).values[train_index]
        yTest = df['popularity'].values[test_index]
        XTest = df.drop(labels='popularity', axis=1).values[test_index]
        yPred = BCModel.predict(XTrain)
        train1_report = classification_report(yTrain, yPred)
        yPred = BCModel.predict(XTest)
        test1_report = classification_report(yTest, yPred)
        BC_accuracy = accuracy_score(yTest, yPred)
        model1_recall = recall_score(yTest, yPred)
        model1_f1 = f1_score(yTest, yPred)
        yPred = RFModel.predict(XTrain)
        train3_report = classification_report(yTrain, yPred)
        yPred = RFModel.predict(XTest)
        test3_report = classification_report(yTest, yPred)
        RF_accuracy = accuracy_score(yTest, yPred)
        model3_recall = recall_score(yTest, yPred)
        model3_f1 = f1_score(yTest, yPred)

    else:
        # Trains Bagging and Random Forest classifiers using warm start.
        yTrain = df['popularity'].values[test_index]
        XTrain = df.drop(labels='popularity', axis=1).values[test_index]  # Creates training data.
        BCModel.fit(XTrain, yTrain)
        RFModel.fit(XTrain, yTrain)
        BCModel.n_estimators += 100 # Must increase n_estimators every iteration.
        RFModel.n_estimators += 100

    i += 1

# Prints all statistics of ensemble classifiers using warm start.
print('Accuracy for the Bagging Classifier using warm start = ', BC_accuracy, '\n')
print('Recall = ', model1_recall, 'F1 = ', model1_f1)
print(test1_report)
print(train1_report)
print('Accuracy for the Random Forest Classifier using warm start = ', RF_accuracy, '\n')
print('Recall = ', model3_recall, 'F1 = ', model3_f1)
print(test3_report)
print(train3_report)

# ***** The rest of the following code is meant to train our best models using the new alternative genre to train the
# model to prove our model can be trained using different genres. ***************************************************
print("*********************Adding alternative genre**************************")
df_og = df.loc[:, ~df.columns.isin(['time_signature'])]
df_alt = pd.read_csv('billboard/alt-songs-billboard.csv')
df_count = pd.read_csv('billboard/country-songs-billboard.csv')
df_pop = pd.read_csv('billboard/pop-songs-billboard.csv')
df_rap = pd.read_csv('billboard/rap-songs-billboard.csv')

#new data; fewer genres, one different
frames2 = [df_pop, df_count, df_rap, df_alt]
genres2 = ['Pop', 'Country', 'Rap', 'Alternative']
genre_nums = [4,2,5,9]
df_new = pd.concat(frames2, ignore_index=True)
df_new.drop_duplicates(inplace=True)

df_new.drop(['Unnamed: 0',
         'disc_number',
         'id', 'is_local',
         'is_playable',
         'name',
         'album_name',
         'track_number',
         'type_x',
         'album_type'],
        axis=1,
        inplace=True)

# replace categorical variables with integers
df_new['Genre'].replace(genres2, genre_nums, inplace=True)
df_new['explicit']. replace([True,False], [1,0], inplace=True)

df_new.rename(columns={"Genre": "genre"}, inplace=True)

df_new.loc[df_new['popularity'] < avg_pop, 'popularity'] = 0
df_new.loc[df_new['popularity'] >= avg_pop, 'popularity'] = 1

df = pd.concat([df_og, df_new], ignore_index=True)

kf = KFold(n_splits=5, shuffle=True)
kfSplit = kf.split(df)

# Creates Bagging, AdaBoost, and Random Forest classifiers along with accuracy arrays.
BCModel = BaggingClassifier(n_estimators=100, bootstrap=True, oob_score=True)
model1_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model1_recall = 0.0
model1_f1 = 0.0
ABModel = AdaBoostClassifier(n_estimators=100)
model2_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model2_recall = 0.0
model2_f1 = 0.0
RFModel = RandomForestClassifier(criterion='gini', n_estimators=100, bootstrap=True, oob_score=True, warm_start=False)
model3_acc = np.array([m.inf, m.inf, m.inf, m.inf, m.inf])
model3_recall = 0.0
model3_f1 = 0.0
# Variables to hold test and train reports for classifiers.
test1_report = 0
train1_report = 0
test2_report = 0
train2_report = 0
test3_report = 0
train3_report = 0

i = 0
# Iterates through different splits.
for train_index, test_index in kfSplit:
    yTrain = df['popularity'].values[train_index]
    XTrain = df.drop(labels='popularity', axis=1).values[train_index] # Creates training data.
    yTest = df['popularity'].values[test_index]
    XTest = df.drop(labels='popularity', axis=1).values[test_index] # Creates testing data.
    BCModel.fit(XTrain, yTrain) # Fits Bagging model.
    yPred = BCModel.predict(XTrain)
    train1_report = classification_report(yTrain, yPred)
    yPred = BCModel.predict(XTest) # Predicts testing data class.
    test1_report = classification_report(yTest, yPred)
    model1_acc[i] = accuracy_score(yTest, yPred) # Accuracy of bagging model on testing data.
    model1_recall += recall_score(yTest, yPred) # recall of bagging model on testing data.
    model1_f1 += f1_score(yTest, yPred) # F1 of bagging model on testing data.
    ABModel.fit(XTrain, yTrain) # Fits AdaBoost model.
    yPred = ABModel.predict(XTrain)
    train2_report = classification_report(yTrain, yPred)
    yPred = ABModel.predict(XTest) # Predicts class of testing data.
    test2_report = classification_report(yTest, yPred)
    model2_acc[i] = accuracy_score(yTest, yPred) # Accuracy of AdaBoost model on testing data.
    model2_recall += recall_score(yTest, yPred)
    model2_f1 += f1_score(yTest, yPred)
    RFModel.fit(XTrain, yTrain) # Creates Random Forest Model.
    yPred = RFModel.predict(XTrain)
    train3_report = classification_report(yTrain, yPred)
    yPred = RFModel.predict(XTest)
    test3_report = classification_report(yTest, yPred)
    model3_acc[i] = accuracy_score(yTest, yPred) # Statistics of Random Forest classifier.
    model3_recall += recall_score(yTest, yPred)
    model3_f1 += f1_score(yTest, yPred)
    i += 1

# Prints all statistics of all 3 ensemble classifiers.
print("Accuracies for Bagging Classifier model using 5 Fold Validation: ", model1_acc)
print("Average accuracy for bagging Classifier = ", np.average(model1_acc))
print("Average recall = ", model1_recall / 5.0, "Average f1 = ", model1_f1 / 5.0)
print("Classification report: \n", test1_report)
print(train1_report)
print("Standard deviation for bagging classifier = ", model1_acc.std(), "\n")
print("Accuracies for AdaBoost Classifier model using 5 Fold Validation: ", model2_acc)
print("Average accuracy for AdaBoost Classifier model = ", np.average(model2_acc))
print("Average recall = ", model2_recall / 5.0, "Average f1 = ", model2_f1 / 5.0)
print("Classification report: \n", test2_report)
print(train2_report)
print("Standard deviation for AdaBoost Classifier model = ", model2_acc.std(), "\n")
print("Accuracies for Random Forest Classifier model using 5 Fold Validation: ", model3_acc)
print("Average accuracy for Random Forest Classifier model = ", np.average(model3_acc))
print("Average recall = ", model3_recall / 5.0, "Average f1 = ", model3_f1 / 5.0)
print("Classification report: \n", test3_report)
print(train3_report)
print("Standard deviation for Random Forest Classifier model = ", model3_acc.std(), "\n")'''
