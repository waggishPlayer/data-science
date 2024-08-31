import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('waggishPlayer/Data-Science/Devtown/capstone-project/movie-recommendation/main_data.csv')
# printing the first 5 rows of the dataframe
movies_data.head()
# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)
# replacing the null valuess with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)
# converting the text data into feature vector
# first create the instance of TfidfVectorizer
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)
print(feature_vectors)
# cosine similarity
#getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)
movie_name=input('Enter your favourite movie name:')
# creating a list with all the movie names given in the dataset
list_of_all_movie_titles=movies_data['title'].tolist()
print(list_of_all_movie_titles)
#finding the close match  for the movie name given by the user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_movie_titles)
print(find_close_match)
close_match=find_close_match[0]
print(close_match)
