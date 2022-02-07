import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def csv_prep():
    movies = pd.read_csv("movies_metadata.csv", usecols=[5, 9, 20])
    movies['index'] = [i for i in range(0, len(movies))]
    return movies


# Load sentences & embeddings from disc
with open('sentence_embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['overviews']
    stored_embeddings = stored_data['sentence_embeddings']


# Get title of movie
def get_title(index):
    return movies[movies.index == index]["title"].values[0]


# Get index of movie
def get_index(title):
    return movies[movies.title == title]["index"].values[0]


def get_similarity():
    return cosine_similarity(stored_embeddings)


def recommend_movie(similarity):
    searching = True
    while searching:
        user_movie = input("Enter the movie for which you want recommendations: ")  # Generate Recommendations
        recommendations = sorted(list(enumerate(similarity[get_index(user_movie)])), key=lambda x: x[1], reverse=True)
        print("The top 3 recommendations for" + " " + user_movie + " " + "are: ")
        print(get_title(recommendations[1][0]), get_title(recommendations[2][0]), get_title(recommendations[3][0]),
              sep="\n")
        decision = input("Press 1 to enter another movie, 0 to exit: ")
        if int(decision) == 0:
            print("Enjoy the film!")
            searching = False


if __name__ == "__main__":
    movies = csv_prep()
    similarity = get_similarity()
    recommend_movie(similarity)
