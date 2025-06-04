# System rekomendacji filmów

Projekt demonstruje prosty system rekomendacji oparty o filtrację współdzieloną. W repozytorium znajdują się dwa skrypty:

- **`rekomendacja.py`** – przykład działania algorytmu na sztucznie wygenerowanych ocenach. Skrypt tworzy losowy zestaw ratingów użytkowników, buduje macierz ocen i wyznacza podobnych użytkowników.
- **`z danymi/rekomendacja_z_danymi.py`** – implementacja korzystająca z rzeczywistych danych pobranych z bazy [MovieLens](https://grouplens.org/datasets/movielens/). Tutaj do obliczeń wykorzystywane są pliki `ratings.csv` oraz `movies.csv`.

## Przebieg działania

1. **Przygotowanie danych**
   - W `rekomendacja.py` dane są generowane losowo. Po utworzeniu DataFrame zapisywany jest plik `ratings.csv` oraz macierz ocen `macierz.csv`.
   - W `rekomendacja_z_danymi.py` dane są wczytywane z `z danymi/ratings.csv`. Każdy wiersz zawiera identyfikator użytkownika (`userId`), identyfikator filmu (`movieId`), ocenę oraz znacznik czasu (`timestamp`). Dodatkowe informacje o filmach znajdują się w `z danymi/movies.csv`.

2. **Budowa macierzy ocen**
   - Z pliku z ocenami tworzona jest macierz gdzie wiersze odpowiadają użytkownikom, kolumny filmom, a wartości to wystawione oceny. Braki zastępowane są zerami.

3. **Wyznaczenie podobieństwa między użytkownikami**
   - Korzystając z funkcji `pairwise_distances` z biblioteki `scikit-learn` obliczany jest współczynnik korelacji między parami użytkowników. Wartości na przekątnej są zerowane, by pomijać porównanie użytkownika z samym sobą.
   - Z uzyskanej macierzy podobieństwa wyprowadzana jest macierz odległości, która następnie służy do wyszukania najbliższych sąsiadów.

4. **Sąsiedzi i predykcja ocen**
   - Model `NearestNeighbors` wyszukuje listę użytkowników najbardziej podobnych do wskazanego użytkownika. W przykładach analizowany jest użytkownik o ID=5.
   - Dla wybranych filmów obliczana jest przewidywana ocena jako średnia ważona ocen sąsiadów (im mniejsze podobieństwo, tym mniejsza waga).
   - Najwyższa przewidywana ocena wskazuje film rekomendowany dla danego użytkownika.

## Informacje o danych

Dane znajdują się w katalogu `z danymi/`:

- **`ratings.csv`** – zawiera wiersze w formacie `userId,movieId,rating,timestamp`. Jest to fragment zbioru MovieLens. Dla przyspieszenia działania można usunąć część wierszy.
- **`movies.csv`** – mapa identyfikatorów filmów na ich tytuły.

Pliki te stanowią podstawę do budowy macierzy ocen i wyznaczenia rekomendacji. W razie problemów z długim czasem obliczeń można ograniczyć liczbę ocen lub filmów w pliku `ratings.csv`.

## Uruchamianie

1. Zainstaluj wymagane biblioteki (numpy, pandas, scikit-learn).
2. W katalogu głównym uruchom `python rekomendacja.py` lub `python z\ danymi/rekomendacja_z_danymi.py`.

Pierwszy skrypt demonstruje działanie na losowych danych, a drugi wykorzystuje rzeczywisty zestaw ocen.

