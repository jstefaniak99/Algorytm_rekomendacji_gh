import numpy
import pandas
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# Generowanie losowych ocen użytkowników
n_users = 1000
n_movies = 100
ratings = pandas.DataFrame({
    'userId': numpy.random.randint(1, n_users, size=10000),
    'movieId': numpy.random.randint(1, n_movies, size=10000), # Losowe oceny użytkowników (wynik zawsze będzie losowy, z każdą kompilacją kodu)
    'rating': numpy.random.randint(1, 11, size=10000)
})

ratings.to_csv('ratings.csv', index=False)

# Tworzenie macierzy userId | movieId
macierz = ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
macierz.to_csv('macierz.csv')

# Obliczanie podobieństwa między użytkownikami
Similar = 1 - pairwise_distances(macierz, metric='correlation')

numpy.fill_diagonal(Similar, 0) # dla tego samego użytkownika ustawiam 0, aby wyeliminować podobieństwo między tymi samymi osobami

# Transformacja macierzy podobieństwa na odległości
max_similarity = numpy.max(Similar)
macierz_odleglosci = max_similarity - Similar
numpy.fill_diagonal(macierz_odleglosci, 0)

# Użycie NearestNeighbors do znalezienia najbardziej podobnych użytkowników
n_neighbors = 10
model = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
model.fit(macierz_odleglosci)
user_id_to_check = 5
neighbours = model.kneighbors([macierz_odleglosci[user_id_to_check]])[1][0][1:]

# Indeksy kolejnych użytkowników w porządku malejącym podobieństwa
most_similar_users_indices = numpy.argsort(Similar[user_id_to_check])[::-1]

# Wybierzmy n-te sąsiedztwo użytkownika
n_neighbours = 10
neighbour_indices = most_similar_users_indices[1:n_neighbours+1]  # Pomijamy samego użytkownika

# Wyświetlmy indeksy najbardziej podobnych użytkowników
print("10 najbardziej podobnych użytkowników dla użytkownika", user_id_to_check, ": ", neighbours)
print("Macierz podobieństwa między użytkownikami:")
print(Similar)

# Zapisywanie macierzy podobieństwa do pliku tekstowego
with open('macierz_podobienstwa.txt', 'w') as file:
    for row in Similar:
        file.write(' '.join([str(elem) for elem in row]) + '\n')

# Ustalamy identyfikator użytkownika, dla którego chcemy przewidzieć oceny
user_id_to_predict = 5

# Wybieramy filmy, dla których chcemy przewidzieć oceny dla danego użytkownika
movies_to_predict = [1, 2, 3, 4, 5]  
predicted_ratings = [] # lista do przechowywania danych

# Iterujemy przez każdy film, dla którego chcemy przewidzieć ocenę
for movie_id in movies_to_predict:
    ratings_for_movie = []
    for neighbor_index in neighbours:
        if macierz.loc[neighbor_index, movie_id] != 0:
            weight = 1 - Similar[user_id_to_predict][neighbor_index]
            ratings_for_movie.append(macierz.loc[neighbor_index, movie_id] * weight)
    
    # Obliczamy przewidywaną ocenę jako średnią ważoną ocen całego sąsiedztwa użytkownika dla danego filmu
    if ratings_for_movie:
        weighted_average_rating = sum(ratings_for_movie) / len(ratings_for_movie)
    else:
        weighted_average_rating = 0
    predicted_ratings.append((movie_id, weighted_average_rating))

# Przewidywane oceny dla wybranych filmów
print("Przewidywane oceny dla użytkownika", user_id_to_predict, "dla wybranych filmów:")
for movie_id, rating in predicted_ratings:
    print("Film ID:", movie_id, "- Przewidywana ocena:", rating)

predicted_ratings.sort(key=lambda x: x[1], reverse=True)
recommended_movie = predicted_ratings[0]
print("Rekomendowany film dla użytkownika", user_id_to_predict, "to:")
print("Film ID:", recommended_movie[0], "- Przewidywana ocena:", recommended_movie[1])