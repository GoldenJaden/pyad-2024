import pandas as pd
from linreg_rec import title_preprocessing

def preprocess_books_for_rec(books_df):
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df = books_df[books_df['Year-Of-Publication'] <= pd.Timestamp.now().year].copy()
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

def recommend_books(user_id, ratings_df, books_df, svd_model, linreg_model, tfidf):
    user_zero_ratings = ratings_df[(ratings_df['User-ID'] == user_id) & (ratings_df['Book-Rating'] == 0)]
    zero_isbns = user_zero_ratings['ISBN'].unique()

    svd_predictions = []
    for isbn in zero_isbns:
        try:
            svd_predictions.append((isbn, svd_model.predict(user_id, isbn).est))
        except Exception as e:
            print(f"Error predicting for ISBN {isbn}: {e}")

    good_books = [(isbn, pred) for isbn, pred in svd_predictions if pred >= 8]
    good_books_isbns = [isbn for isbn, _ in good_books]

    books_features = books_df[books_df['ISBN'].isin(good_books_isbns)]
    books_features['Book-Title'] = books_features['Book-Title'].apply(title_preprocessing)

    tfidf_vectors = tfidf.transform(books_features['Book-Title'])

    linreg_predictions = linreg_model.predict(tfidf_vectors)

    recommendations = pd.DataFrame({
        'ISBN': books_features['ISBN'],
        'Title': books_features['Book-Title'],
        'SVD_Pred': [pred for _, pred in good_books],
        'LinReg_Pred': linreg_predictions
    })
    recommendations.sort_values(by='LinReg_Pred', ascending=False, inplace=True)

    return recommendations[['Title', 'SVD_Pred', 'LinReg_Pred']]

# Recommendations for user 198711:
#                                                      Title  SVD_Pred  LinReg_Pred
# 19015                 The Monster at the End of This Book  8.035243     7.626934
# 3354    The Hobbit : The Enchanting Prelude to The Lor...  8.011301     7.612899
# 5601    She Said Yes : The Unlikely Martyrdom of Cassi...  8.159763     7.608617
# 4023    Cat in the Hat (I Can Read It All by Myself Be...  8.099796     7.607852
# 1024                                The Phantom Tollbooth  8.231862     7.607227
# 3459     Harry Potter and the Chamber of Secrets (Book 2)  8.033508     7.606293
# 26292   Hop on Pop (I Can Read It All by Myself Beginn...  8.248170     7.603735
# 138066  The Very Best Baby Name Book in the Whole Wide...  8.057127     7.602915
# 21746                           Goodnight Moon Board Book  8.002453     7.602697
# 15307                                       Evening Class  8.124121     7.602062
# 900                        The Magician's Nephew (Narnia)  8.299160     7.601522
# 29663               The Magician's Nephew (rack) (Narnia)  8.300297     7.601522
# 14110   Into Thin Air : A Personal Account of the Moun...  8.391927     7.599969
# 2554                                     The Color Purple  8.194194     7.596809
# 56088                                      A Time to Kill  8.016737     7.596702
# 16780                                   A Wrinkle In Time  8.258615     7.596204
# 1534                                       The Green Mile  8.058735     7.595100
# 4466                                     Love You Forever  8.195297     7.593329
# 19004                            The Biggest Pumpkin Ever  8.466267     7.593116
# 24292   The Lion, the Witch and the Wardrobe (Full-Col...  8.176206     7.592657
# 915                    The Giver (21st Century Reference)  8.545714     7.590393
# 3847                     Charlotte's Web (Trophy Newbery)  8.121488     7.589825
# 304                                 October Sky: A Memoir  8.219019     7.588950
# 4479                                 Silence of the Lambs  8.032841     7.588815
# 2143    Harry Potter and the Sorcerer's Stone (Harry P...  8.158120     7.588718
# 16083                                        The Hot Zone  8.151224     7.588249
# 18747                                             Frindle  8.183582     7.587700
# 1323                                        Lonesome Dove  8.257172     7.587700
# 60659                     Snow White and the Seven Dwarfs  8.324068     7.585547
# 1387                              A Prayer for Owen Meany  8.068714     7.585116