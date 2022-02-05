import pandas as pd

movies = pd.read_csv("movies_metadata.csv", usecols = [5,9,20])


def get_title(index):
    return movies[movies.index == index]["title"].values[0]


def get_index(title):
    return movies[movies.title == title]["index"].values[0]


movies['index'] = [i for i in range(0, len(movies))]
movies = movies.dropna()


