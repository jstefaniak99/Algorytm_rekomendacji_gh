import numpy
import pandas
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

ratings_df = pandas.read_csv('ratings.csv')

# Tworzenie macierzy userId | movieId na podstawie ocen
macierz = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Obliczanie podobieństwa między użytkownikami
Similar = 1 - pairwise_distances(macierz, metric='correlation')

numpy.fill_diagonal(Similar, 0)  # Dla tego samego użytkownika ustawiamy 0, aby wyeliminować podobieństwo między tymi samymi osobami

# Transformacja macierzy podobieństwa na odległości
max_similarity = numpy.max(Similar)
macierz_odleglosci = max_similarity - Similar
numpy.fill_diagonal(macierz_odleglosci, 0)

# Użycie NearestNeighbors do znalezienia najbardziej podobnych użytkowników
n_neighbors = 10
model = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
model.fit(macierz_odleglosci)

# Znalezienie najbardziej podobnych użytkowników dla konkretnego użytkownika
user_id_to_check = 5
neighbours = model.kneighbors([macierz_odleglosci[user_id_to_check]])[1][0][1:]

# Wybieramy filmy, dla których chcemy przewidzieć oceny dla danego użytkownika
movies_to_predict = [1, 2, 3, 4, 5] 
predicted_ratings = [] # lista do przechowywania danych

# Iteracja przez każdy film, dla którego chcemy przewidzieć ocenę
for movie_id in movies_to_predict:
    ratings_for_movie = []
    for neighbor_index in neighbours:
        if macierz.loc[neighbor_index, movie_id] != 0:
            weight = 1 - Similar[user_id_to_check][neighbor_index]
            ratings_for_movie.append(macierz.loc[neighbor_index, movie_id] * weight)
    
    # Obliczanie przewidywanej oceny jako średniej ważonej ocen całego sąsiedztwa użytkownika dla danego filmu
    if ratings_for_movie:
        weighted_average_rating = sum(ratings_for_movie) / len(ratings_for_movie)
    else:
        weighted_average_rating = 0
    predicted_ratings.append((movie_id, weighted_average_rating))

# Sortowanie przewidywanych ocen w kolejności malejącej
predicted_ratings.sort(key=lambda x: x[1], reverse=True)

# Wybieranie filmu z najwyższą przewidywaną oceną
recommended_movie = predicted_ratings[0]
print("Rekomendowany film dla użytkownika", user_id_to_check, "to:")
print("Film ID:", recommended_movie[0], "- Przewidywana ocena:", recommended_movie[1])
