README

______________________________________________________________________
PROJECT
Name: Popularity Classification Analysis of Spotify Tracks
Class: CS 488
Semester: Fall 2022
Group: Kerry Forsythe, Roderick Hutterer, David Torres

______________________________________________________________________
PROJECT STAGES
Stage 1 - Group identification
Stage 2 - Problem definition
Stage 3 - Data collection, pre-processing and descriptive analytics
Stage 4 - Classification Analysis
Stage 5 - Reporting

______________________________________________________________________
ORGANIZATION -  how the code base is organized
billboard - CSV files for billboard chart Spotify Songs generated in stage5-billboard-data.ipynb
billboard_summary_stats.csv - summary statistics for Billboard dataset, used in stage 5
full_dataset.csv - assembled Spotify dataset with pre-processing
spotify_data - folder with 8 original CSV datasets generated in stage3_spotify_data.ipynb
stage3_spotify_data.ipynb - Jupyter Notebooks file for gathering track data for 8 genres from Spotify
stage3_spotify_analysis.py - initial pre-processing of data files and descriptive analytics
stage4_classification.py - classification analysis of Spotify data
stage4-model-compare-plot.py - python file for generating a comparison plot for models using results from a specific run
stage5-billboard-analysis.py - Random forest classification trained on the original dataset, tested on Billboard dataset
roderick_analysis_stage5.py - Has extra code for Naive Bayes Model, and ensemble classifiers using warm start. Also retrains models using new genre.
stage5-billboard-data.ipynb - Jupyter Notebooks file for gathering track data from several Billboard charts
report.pdf - final project report
stratified_sample.csv - example of a stratified sample of the full dataset
summary_stats.csv - summary statistics for full dataset
visualizations - folder with PNG visualizations generated from code

______________________________________________________________________
COMMANDS - the commands to run your code
To run python files, type python3 followed by name of the file.
Example: python3 stage3_spotify_analysis.py
To run notebook files, open in Jupyter Notebooks or Google Colaboratory, and click run.

Jupyter Notebooks is part of the Anaconda Distribution available at https://www.anaconda.com/products/distribution

Google Colaboratory is available at https://colab.research.google.com/

______________________________________________________________________
PACKAGES/LIBRARIES - names and versions of all the packages/libraries
billboard 7.0.0
numpy 1.21.5
pandas 1.4.2
matplotlib 3.5.1
seaboarn 0.11.2
scipy 1.7.3
spotipy 2.20.0

______________________________________________________________________
DATASET - data set information
The datasets were gathered using the spotipy package (https://spotipy.readthedocs.io/en/2.21.0/) to connect to the Spotify data api.  See stage2_spotify_data.ipynb file for code.
