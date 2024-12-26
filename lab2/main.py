import pandas as pd
from linreg_rec import preprocess_books, preprocess_ratings, train_svd, train_regressor
from personal_recommendation import get_user_with_most_zero_ratings, recommend_books, preprocess_ratings_for_rec, preprocess_books_for_rec
import pickle
from os.path import isfile

def main():
    books_df = pd.read_csv('./Books.csv')
    ratings_df = pd.read_csv('./Ratings.csv')

    if not(isfile('./svd_model.pkl')) or not(isfile('./linreg.pkl')):
        # Препроцессинг
        preprocessed_books_df = preprocess_books(books_df)
        preprocessed_ratings_df = preprocess_ratings(ratings_df)

        train_svd(preprocessed_ratings_df)
        train_regressor(preprocessed_books_df, preprocessed_ratings_df)

    preprocessed_books_df = preprocess_books_for_rec(books_df)
    preprocessed_ratings_df = preprocess_ratings_for_rec(ratings_df)

    # Загрузка моделей
    with open('svd_model.pkl', 'rb') as file:
        svd_model = pickle.load(file)

    with open('linreg.pkl', 'rb') as file:
        linreg_data = pickle.load(file)
        linreg_model = linreg_data['model']
        tfidf = linreg_data['tfidf']
        encoder = linreg_data['encoder']
        scaler = linreg_data['scaler']

    user_id = get_user_with_most_zero_ratings(ratings_df)

    recommendations = recommend_books(user_id, preprocessed_ratings_df, preprocessed_books_df, svd_model, linreg_model, tfidf, encoder, scaler)

    print(f"Recommendations for user {user_id}:\n", recommendations)

if __name__ == "__main__":
    main()