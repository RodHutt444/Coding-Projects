import spotipy as spy
from spotipy_random import get_random
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import pandas as pd
import numpy as np
import random as r

CLIENT_ID = '6967778a64e94056912cdb0f50e63811'
CLIENT_SECRET = '527e8c338e434fb9913a368fcf6cd91c'
PLAYLIST_URI = 'spotify:playlist:3L9ixpoC6iAdyjyCJSIwoc'

scope = "user-library-read"

spc = spy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
spa = spy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

def get_random_tracks(rs):
    count = 0
    while count <= 250:
        track = get_random(spa, type='track')
        track_name = track['name']
        track_popularity = spc.track(track['id'])['popularity']
        rs = pd.concat([rs, pd.DataFrame([{'name': track_name, 'popularity': track_popularity} |
                                          spc.audio_features(track['uri'])[0]])], ignore_index=True)
        count += 1
        print(count)
    rs = rs.drop_duplicates()
    return rs


'''def get_tracks():
    results = spc.playlist_tracks(PLAYLIST_URI)
    tracks = results['items']
    while results['next']:
        results = spc.next(results)
        tracks.extend(results['items'])
    return tracks

playlist_tracks = get_tracks()
df = pd.DataFrame()

count = 0
for track in playlist_tracks:
    track_uri = track['track']['uri']
    track_name = track['track']['name']
    df = pd.concat([df, pd.DataFrame([{'name': track_name, 'popularity': spc.track(track['track']['id'])['popularity']} |
                                      spc.audio_features(track_uri)[0]])], ignore_index=True)

df.to_csv('popularSongs.csv', index=False)'''

df = pd.DataFrame()
df = get_random_tracks(df)
print("Number of songs: ", len(df))
while len(df) < 1000:
    df = get_random_tracks(df)
    print("Number of songs: ", len(df))
df.to_csv('unpopularSongs.csv', index=False)
