import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder

def preprocess_books(books_df):
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df = books_df[books_df['Year-Of-Publication'] <= pd.Timestamp.now().year]
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

def train_svd(ratings_df):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
    svd = SVD()

    results = cross_validate(svd, data, measures=['MAE'], cv=5, verbose=True)
    mean_mae = np.mean(results['test_mae'])
    print(f'Mean MAE for SVD: {mean_mae}')

    if mean_mae > 1.3:
        print("Warning: MAE for SVD is above 1.3")
    else:
        print("SVD training successful with MAE below 1.3")

    with open('svd_model.pkl', 'wb') as file:
        pickle.dump(svd, file)

def train_regressor(books_df, ratings_df):
    merged_df = ratings_df.merge(books_df, on='ISBN')

    tfidf = TfidfVectorizer(max_features=300)
    title_vectors = tfidf.fit_transform(merged_df['Book-Title'])

    cat_features = merged_df[['Book-Author', 'Publisher']]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    cat_sparse = encoder.fit_transform(cat_features)

    year_data = merged_df[['Year-Of-Publication']].astype(float)
    scaler = StandardScaler()
    year_data_scaled = scaler.fit_transform(year_data.values)
    year_sparse = csr_matrix(year_data_scaled)

    X_sparse = hstack([title_vectors, cat_sparse, year_sparse], format='csr')

    y = merged_df['Book-Rating']

    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

    reg = SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    y_pred_clamped = np.clip(y_pred, 1, 10)

    mae = mean_absolute_error(y_test, y_pred_clamped)
    print(f'MAE for linear regression: {mae}')

    if mae > 1.5:
        print("Warning: MAE for linear regression is above 1.5")
    else:
        print("Linear regression training successful with MAE below 1.5")

    with open('linreg.pkl', 'wb') as file:
        pickle.dump({'model': reg, 'tfidf': tfidf, 'encoder': encoder, 'scaler': scaler}, file)
