import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd


print("Reading csv")
movies = pd.read_csv("movies_metadata.csv", usecols=[5, 9, 20])
print("Finished reading csv")


def clean_data(dataset):
    dataset['index'] = [i for i in range(0, len(movies))]
    dataset = dataset.dropna()
    return dataset


clean_movies = clean_data(movies)

print("Creating bert")
bert = SentenceTransformer('all-MiniLM-L6-v2')
print("embedding sentences")
sentence_embeddings = bert.encode(clean_movies['overview'].tolist(), show_progress_bar = True)

with open('sentence_embeddings.pkl', 'wb') as fOut:
    pickle.dump({'overviews': clean_movies['overview'], 'sentence_embeddings': sentence_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

