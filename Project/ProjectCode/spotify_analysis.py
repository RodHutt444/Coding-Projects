#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Kerry Forsythe, Roderick Hutterer, David Torres
Course: CS 488
Semester: Fall 2022
Project: Group Project
Filename: spotify_analysis.py
"""

#import packages
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#show all the columns in data set when printing
pd.set_option('display.max_columns', None)

#separator for printing between questions
sep = "\n----------------------------------------------------\n"

# -------------------------------------------------------
# CREATE DATAFRAMES
# read in data sets
df_pop = pd.read_csv('pop_songs_feat.csv')
df_count = pd.read_csv('country_songs_feat.csv')
df_rock = pd.read_csv('rock_songs_feat.csv')
df_rap = pd.read_csv('rap_songs_feat.csv')
df_jazz = pd.read_csv('jazz_songs_feat.csv')
df_class = pd.read_csv('classical_songs_feat.csv')
df_world = pd.read_csv('world_songs_feat.csv')
df_techno = pd.read_csv('techno_songs_feat.csv')

#frames and genres
frames = [df_pop, df_count, df_rock, df_rap, df_class, df_jazz, df_world,df_techno]
genres = ['pop', 'country', 'rock', 'rap', 'classical', 'jazz', 'world', 'techno']
genre_nums = [1,2,3,4,5,6,7,8]

# merge datasets
df = pd.concat(frames, ignore_index=True)
print(df.columns)

#drop some unneeded columns
df.drop(['disc_number','is_local', 'is_playable', 'track_number', 'type_x', 'linked_from'], axis=1, inplace=True)

df.to_csv("all_genres_data_set.csv")
#print suummary statistics
print(sep, "Overall statistic summary")


#replace some categorical text variables with integers
df['explicit'].replace([True, False],
                        [1, 0], inplace=True)

df["album_type"].replace(["single", "album", "compilation"], [0,1,2], inplace=True)

df['genre_numeric'] = df["genre"].replace(
    genres,
    genre_nums)

#select only numeric columns
df_num = df.select_dtypes(np.number)

# -------------------------------------------------------
# SUMMARY STATISTICS
#print summary satistics
#overall
print(df.describe())
#by genre
for genre in genres:
    print(sep, genre)
    print(df[df['genre'] == genre].describe())


# -------------------------------------------------------
# HISTOGRAMS
# overall
df_num.hist(figsize=(20,17), bins=15)
plt.savefig("histograms/full_data_histogramss.png") 

# by genre
for i in range(1,9):
    df_genre = df_num[df_num['genre_numeric'] == i]
    df_genre.hist(figsize=(20,17), bins=15)
    plt.savefig("histograms/" + genres[i-1] + "_histograms.png")

# -------------------------------------------------------
# Boxplots
for col in df_num.columns:
    sns.set(rc={"figure.figsize":(12, 8)})
    plot_title = col + " Box Plot"
    sns.boxplot(x="genre", y=col, data=df).set(title=plot_title)
    plt.savefig("boxplots/" + col+"_box_plot.png")
    plt.show()


