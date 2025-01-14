import pandas as pd
from linreg_rec import preprocess_books, preprocess_ratings, train_regressor, prepare_data, transform_data
from svd_rec import modeling
from personal_recommendation import get_user_with_most_zero_ratings, recommend_books, preprocess_ratings_for_rec, preprocess_books_for_rec
import pickle
from os.path import isfile
import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download('punkt_tab')

def main():
    books_df = pd.read_csv('./Books.csv', low_memory=False)
    ratings_df = pd.read_csv('./Ratings.csv')

    # Препроцессинг
    preprocessed_books_df = preprocess_books(books_df)
    preprocessed_ratings_df = preprocess_ratings(ratings_df)

    data = prepare_data(preprocessed_ratings_df, preprocessed_books_df)

    X, y, tfidf = transform_data(data)

    train_regressor(X, y)
    modeling(preprocessed_ratings_df)

    preprocessed_books_df = preprocess_books_for_rec(books_df)
    preprocessed_ratings_df = preprocess_ratings_for_rec(ratings_df)

    # Загрузка моделей
    with open('svd_model.pkl', 'rb') as file:
        svd_model = pickle.load(file)

    with open('linreg.pkl', 'rb') as file:
        linreg_model = pickle.load(file)

    user_id = get_user_with_most_zero_ratings(ratings_df)

    recommend_books(user_id, preprocessed_ratings_df, preprocessed_books_df, svd_model, linreg_model, tfidf)

if __name__ == "__main__":
    main()