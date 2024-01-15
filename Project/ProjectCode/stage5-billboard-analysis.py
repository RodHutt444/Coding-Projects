#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Kerry Forsythe, Roderick Hutterer, David Torres
Course: CS 488
Semester: Fall 2022
Project: Group Project
Filename: billboard_analysis.py
"""
import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#show all the columns in data set when printing
pd.set_option('display.max_columns', None)

sep = "\n----------------------------------------------------\n"


#---------------------------------------------------------------------------
#ORIGINAL DATASET
df_og = pd.read_csv('full_dataset.csv')

#replace categorical variables with numeric
genres = ['classical', 'country', 'jazz', 'pop', 'rap', 'rock', 'techno', 'world']
genres_num = [1,2,3,4,5,6,7,8]
df_og['genre'].replace(genres, genres_num, inplace=True)
df_og['explicit']. replace([True,False], [1,0], inplace=True)


#get mean, median, standard deviation
mean_pop = df_og['popularity'].mean()
median_pop = df_og['popularity'].median()
std_pop = df_og['popularity'].std()


print(df_og['popularity'].describe())
og_pop = df_og['popularity'].copy()

# Popularity Binary Classification 
df_og.loc[df_og['popularity'] < mean_pop + std_pop,'popularity'] = 0
df_og.loc[df_og['popularity'] >= mean_pop + std_pop, 'popularity'] = 1

#drop unneeded columns, Note: time signature not in 2nd dataset
df_og.drop(['is_local', 'name', 'time_signature'], axis=1, inplace=True)

#stratifiedSample = df_og.groupby('genre', group_keys=False).apply(lambda x: x.sample(frac=0.1))
#df_og = stratifiedSample

#correlation matrix
pMatrix = df_og.corr()
print(pMatrix)

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(pMatrix,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200)
            )

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize = 18,
    horizontalalignment='right'
);
ax.set_yticklabels(ax.get_yticklabels(),fontsize = 18)
plt.savefig('corr_heat.png')


#---------------------------------------------------------------------------
# NEW DATASET
#get data
df_alt = pd.read_csv('billboard/alt-songs-billboard.csv')
df_count = pd.read_csv('billboard/country-songs-billboard.csv')
df_pop = pd.read_csv('billboard/pop-songs-billboard.csv')
df_rap = pd.read_csv('billboard/rap-songs-billboard.csv')

#new data; fewer genres, one different
frames2 = [df_pop, df_count, df_rap, df_alt, df_og]
genres2 = ['Pop', 'Country', 'Rap', 'Alternative']
genre_nums = [4,2,5,9]


#merge dataframes
df = pd.concat(frames2, ignore_index=True)
print(df.columns)

#check for duplicates
df.drop_duplicates(inplace=True)

stats = df.describe()
#print (stats)
# stats.to_csv('billboard_summary_stats.csv')

#drop some unneeded columns
df.drop(['Unnamed: 0',
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
df['Genre'].replace(genres2, genre_nums, inplace=True)
df['explicit']. replace([True,False], [1,0], inplace=True)

df.rename(columns={"Genre": "genre"}, inplace=True)

# df.hist(figsize=(20,17), bins=15)
# plt.savefig("visualizations/histograms/billboard_data_histogramss.png")


#use original dataset values for splitting #s
df.loc[df['popularity'] < mean_pop + std_pop,'popularity'] = 0
df.loc[df['popularity'] >= mean_pop + std_pop, 'popularity'] = 1

#reorder columns to match orginal dataset
df = df.iloc[:,[0,1,2,3,15,4,5,6,7,8,9,10,11,12,13,14]]

#correlation matrix
pMatrix = df.corr()
# print(pMatrix)

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(pMatrix,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            )

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize = 18,
    horizontalalignment='right'
);
ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize = 18
    )
plt.title('Heatmap of Billboard Correlation Matrix', fontsize = 25) 
# plt.savefig('visualizations/heatmap/billboard_corr_heat.png')






# ------------------------------------------------------------------------------
# Training - original dataset
st = time.time()
train, test = train_test_split(df,test_size=0.2, random_state=0)
# get X and y
XTrain = train.drop(["popularity"],axis=1)
XTest = test.drop(["popularity"], axis=1)
yTrain = train["popularity"]
yTest = test["popularity"]
feature_list = list(XTrain.columns)
et = time.time() - st

# ------------------------------------------------------------------------------
#Random Forest - train & test on original data
model = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, oob_score=True, warm_start=False, max_depth=None, max_samples=None)
model_acc = []

model.fit(XTrain, yTrain)
yPred = model.predict(XTest)
yPred_train = model.predict(XTrain)
model_acc.append(accuracy_score(yTest, yPred))
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

print(f'{sep}Full Data Set - Random Forest')
print("Accuracies for Random Forest Test: ", accuracy_score(yTest, yPred))
print("Average accuracy for Random Forest Training ", accuracy_score(yTrain, yPred_train))
print("Elapsed time",et, "\n")
print("Training confusion matrix\n", confusion_matrix(yTrain, yPred_train))
print("Testing confusion matrix:\n", confusion_matrix(yTest, yPred))

print("Training classification report:\n", classification_report(yTrain, yPred_train))
print("Test classification report:\n", classification_report(yTest, yPred))

#feature importances
feature_names = XTrain.columns
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize=(15,8))
sorted_idx = model.feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx], xerr=std[sorted_idx])
#orest_importances.plot.barh(xerr=std, ax=ax)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
ax.set_title("Feature importances", fontsize=10)
fig.tight_layout()
# plt.savefig('rf-importances-full.png')
plt.show()
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

df_results = test
df_results['pred_pop'] = yPred
#accuracies by genre
genre_acc = []
for i in range(len(genres_num)):
    subset = df_results[df_results['genre'] == genres_num[i]]
    yTest = subset['popularity']
    yPred = subset['pred_pop']
    acc = accuracy_score(yTest, yPred)
    genre_acc.append(acc)
    print(genres[i],acc)

fig, ax = plt.subplots()
plt.barh(genres, genre_acc)
plt.ylabel("Genre",fontsize=20)
plt.xlabel("Accuracy",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title("Original Spotify Accuraccy by Genre",fontsize=20)
# plt.savefig('visualizations/bar/accuracy_by_genre.png')

# ------------------------------------------------------------------------------
#Random Forest - test on new data
XTest2 = df.drop(['popularity'], axis=1)
yTest2 = df['popularity']

#predict the popularity
yPred2 = model.predict(XTest2)
bill_acc = accuracy_score(yTest2, yPred2)
print(f"{sep}Accuracies for Random Forest Billboard Test: {bill_acc}")
print("Testing confusion matrix:\n", confusion_matrix(yTest2, yPred2))
print("Test classification report:\n", classification_report(yTest2, yPred2))
df_results = df.copy()
df_results['pred_pop'] = yPred2

#accuracies by genre
genre_acc = []

for i in range(len(genre_nums)):
    subset = df_results[df_results['genre'] == genre_nums[i]]
    yTest = subset['popularity']
    yPred = subset['pred_pop']
    acc = accuracy_score(yTest, yPred)
    genre_acc.append(acc)
    print(genres2[i],acc)

fig, ax = plt.subplots(figsize=(15,8))
plt.barh(genres2, genre_acc)
plt.ylabel("Genre", fontsize=20)
plt.xlabel("Accuracy",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title("Billboard Accuraccy by Genre",fontsize=20)
# plt.savefig('visualizations/bar/billboard_accuracy_by_genre.png')

df

