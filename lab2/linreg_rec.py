import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.accuracy import mae
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
import nltk

def title_preprocessing(text: str) -> str:
    # Токенизация
    tokens = nltk.word_tokenize(text)

    # Удаление стоп-слов и пунктуации
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

    return ' '.join(tokens)

def preprocess_books(books_df):
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df = books_df[books_df['Year-Of-Publication'] <= pd.Timestamp.now().year].copy()
    books_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
    books_df.dropna(inplace=True)
    return books_df

def preprocess_ratings(ratings_df):
    ratings_df = ratings_df[ratings_df['Book-Rating'] > 0]
    book_counts = ratings_df['ISBN'].value_counts()
    user_counts = ratings_df['User-ID'].value_counts()
    ratings_df = ratings_df[ratings_df['ISBN'].isin(book_counts[book_counts > 1].index)]
    ratings_df = ratings_df[ratings_df['User-ID'].isin(user_counts[user_counts > 1].index)]
    return ratings_df

def prepare_data(ratings_df, books_df):

    data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')

    avg_ratings = data.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)

    full_data = pd.merge(books_df, avg_ratings, on='ISBN', how='inner')

    full_data.dropna(subset=['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication'], inplace=True)

    return full_data

def transform_data(data):
    tfidf = TfidfVectorizer(max_features=1000)
    data['Book-Title'] = data['Book-Title'].apply(title_preprocessing)
    title_vectors = tfidf.fit_transform(data['Book-Title']).toarray()

    data['Book-Author'] = data['Book-Author'].astype('category').cat.codes
    data['Publisher'] = data['Publisher'].astype('category').cat.codes

    data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')

    X = pd.concat([
        pd.DataFrame(title_vectors, index=data.index),
        data[['Book-Author', 'Publisher', 'Year-Of-Publication']]
    ], axis=1)

    X.columns = X.columns.astype(str)

    y = data['Average-Rating']

    return X, y, tfidf

def train_regressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SGDRegressor(max_iter=2000, tol=1e-4, alpha=0.01, penalty='l2', learning_rate='adaptive')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    if mae < 1.5:
        with open("linreg.pkl", "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved successfully with MAE: {mae}")
    else:
        print(f"Model not saved. MAE is too high: {mae}")
