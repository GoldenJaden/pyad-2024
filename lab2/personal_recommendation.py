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

def preprocess_books_for_rec(books_df):
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df = books_df[books_df['Year-Of-Publication'] <= pd.Timestamp.now().year]
    books_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
    books_df.dropna(inplace=True)
    return books_df

def preprocess_ratings_for_rec(ratings_df):
    book_counts = ratings_df['ISBN'].value_counts()
    user_counts = ratings_df['User-ID'].value_counts()
    ratings_df = ratings_df[ratings_df['ISBN'].isin(book_counts[book_counts > 1].index)]
    ratings_df = ratings_df[ratings_df['User-ID'].isin(user_counts[user_counts > 1].index)]
    return ratings_df

def get_user_with_most_zero_ratings(ratings_df):
    zero_ratings = ratings_df[ratings_df['Book-Rating'] == 0]
    user_id = zero_ratings['User-ID'].value_counts().idxmax()
    print(f"\nПользователь с максимальным количеством нулевых рейтингов: {user_id}")
    print(f"Количество нулевых рейтингов: {zero_ratings.max()}")
    return user_id

def recommend_books(user_id, ratings_df, books_df, svd_model, linreg_model, tfidf, encoder, scaler):
    # Фильтруем книги, которым пользователь поставил 0
    user_zero_ratings = ratings_df[(ratings_df['User-ID'] == user_id) & (ratings_df['Book-Rating'] == 0)]
    zero_isbns = user_zero_ratings['ISBN'].unique()

    # Прогнозируем оценки SVD для этих книг
    svd_predictions = []
    for isbn in zero_isbns:
        svd_predictions.append((isbn, svd_model.predict(user_id, isbn).est))

    # Отбираем книги с прогнозируемым рейтингом SVD >= 8
    good_books = [(isbn, pred) for isbn, pred in svd_predictions if pred >= 8]

    # Преобразуем книги для линейной регрессии
    good_books_isbns = [isbn for isbn, _ in good_books]
    books_features = books_df[books_df['ISBN'].isin(good_books_isbns)]

    tfidf_vectors = tfidf.transform(books_features['Book-Title'])
    cat_features = encoder.transform(books_features[['Book-Author', 'Publisher']])
    year_features = scaler.transform(books_features[['Year-Of-Publication']].astype(float))
    X_sparse = hstack([tfidf_vectors, cat_features, csr_matrix(year_features)], format='csr')

    linreg_predictions = linreg_model.predict(X_sparse)

    recommendations = pd.DataFrame({
        'ISBN': books_features['ISBN'],
        'Title': books_features['Book-Title'],
        'SVD_Pred': [pred for _, pred in good_books],
        'LinReg_Pred': linreg_predictions
    })
    recommendations.sort_values(by='LinReg_Pred', ascending=False, inplace=True)

    return recommendations[['Title', 'SVD_Pred', 'LinReg_Pred']]

# Recommendations for user 198711:

# 3459     Harry Potter and the Chamber of Secrets (Book 2)  8.092041     8.426539
# 2143    Harry Potter and the Sorcerer's Stone (Harry P...  8.063812     8.360164
# 3354    The Hobbit : The Enchanting Prelude to The Lor...  8.168143     8.220500
# 29663               The Magician's Nephew (rack) (Narnia)  8.065026     8.011973
# 3919                                            Yukon Ho!  8.278071     7.968351
# 516                                        The Bean Trees  8.328333     7.956274
# 3847                     Charlotte's Web (Trophy Newbery)  8.335404     7.881777
# 26292   Hop on Pop (I Can Read It All by Myself Beginn...  8.203190     7.866158
# 1534                                       The Green Mile  8.091098     7.830255
# 4479                                 Silence of the Lambs  8.158498     7.820127
# 1024                                The Phantom Tollbooth  8.326480     7.807844
# 1387                              A Prayer for Owen Meany  8.135964     7.798336
# 16198       Key of Valor (Roberts, Nora. Key Trilogy, 3.)  8.332131     7.790570
# 16780                                   A Wrinkle In Time  8.169674     7.785730
# 22950   The Black Cauldron (Chronicles of Prydain (Pap...  8.520010     7.778246
# 10300                                        The Talisman  8.161424     7.777406
# 915                    The Giver (21st Century Reference)  8.069358     7.770885
# 1105      Divine Secrets of the Ya-Ya Sisterhood: A Novel  8.140121     7.751508
# 2554                                     The Color Purple  8.250932     7.743529
# 19004                            The Biggest Pumpkin Ever  8.002433     7.737003
# 1707                  Howl and Other Poems (Pocket Poets)  8.193725     7.730985
# 21746                           Goodnight Moon Board Book  8.408596     7.729772
# 138066  The Very Best Baby Name Book in the Whole Wide...  8.043590     7.720733
# 8468                             A Swiftly Tilting Planet  8.162136     7.691817
# 5601    She Said Yes : The Unlikely Martyrdom of Cassi...  8.168489     7.638232
# 304                                 October Sky: A Memoir  8.022968     7.631253
# 41798   Angels Everywhere: A Season of Angels/Touched ...  8.239926     7.630218
# 8797                      One Flew Over the Cuckoo's Nest  8.122364     7.625198
# 30913                                 Good Night, Gorilla  8.208753     7.608124
# 2941                                            Intensity  8.027285     7.589178
# 7060                             The Hunt for Red October  8.322353     7.579643
# 3747                                                Night  8.350974     7.579299
# 15307                                       Evening Class  8.338346     7.576409
# 1323                                        Lonesome Dove  8.170160     7.538924