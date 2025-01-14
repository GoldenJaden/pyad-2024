import pandas as pd
from linreg_rec import title_preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    user_ratings = ratings_df[ratings_df["User-ID"] == user_id]
    zero_rated_books = user_ratings[user_ratings["Book-Rating"] == 0]["ISBN"].tolist()

    print(
        f"\nКоличество книг с нулевым рейтингом для пользователя {user_id}: {len(zero_rated_books)}"
    )

    # Предсказание SVD
    predictions = []
    for isbn in zero_rated_books:
        prediction = svd_model.predict(uid=user_id, iid=isbn).est
        predictions.append((isbn, prediction))

    # Оставляем только книги с предсказанным рейтингом >= 8
    recommended_books = [(isbn, rating) for isbn, rating in predictions if rating >= 8]

    # Сортируем книги по рейтингу SVD
    recommended_books = sorted(recommended_books, key=lambda x: x[1], reverse=True)

    recommended_books_isbn = [book[0] for book in recommended_books]

    data = pd.merge(ratings_df, books_df, on="ISBN", how="inner")
    data = data[data["ISBN"].isin(recommended_books_isbn)]
    data.dropna(inplace=True)

    le_author = LabelEncoder()
    le_publisher = LabelEncoder()
    data["Book-Author"] = le_author.fit_transform(data["Book-Author"])
    data["Publisher"] = le_publisher.fit_transform(data["Publisher"])
    data["Book-Title"] = data["Book-Title"].apply(title_preprocessing)

    book_titles = tfidf.transform(data["Book-Title"]).toarray()

    scaler = StandardScaler()
    books_scaled = scaler.fit_transform(
        data[["Book-Author", "Publisher", "Year-Of-Publication"]]
    )

    X_linreg = pd.concat(
        [
            data[["ISBN", "Book-Title"]].reset_index(drop=True),
            pd.DataFrame(books_scaled),
            pd.DataFrame(book_titles),
        ],
        axis=1,
    )

    linreg_predictions = linreg_model.predict(X_linreg.iloc[:, 2:])

    X_linreg["Predicted-Rating-LinReg"] = linreg_predictions

    isbn_to_svd_rating = dict(recommended_books)
    X_linreg["Predicted-Rating-SVD"] = X_linreg["ISBN"].map(isbn_to_svd_rating)

    final_recommendations = X_linreg.drop_duplicates(subset=["ISBN"]).sort_values(
        by="Predicted-Rating-LinReg", ascending=False
    )

    print("\nКонечный список рекомендаций:")
    print(final_recommendations[["ISBN", "Book-Title", "Predicted-Rating-SVD", "Predicted-Rating-LinReg"]])

# Конечный список рекомендаций:
#            ISBN                                         Book-Title  Predicted-Rating-SVD  Predicted-Rating-LinReg
# 3    039480029X    Hop Pop I Can Read It All Myself Beginner Books              8.047403                 7.648609
# 39   0836218353                                           Yukon Ho              8.132317                 7.621427
# 8    0064400557                       Charlotte Web Trophy Newbery              8.110368                 7.620690
# 12   0671617028                                   The Color Purple              8.218328                 7.619322
# 19   0064405052                         The Magician Nephew Narnia              8.066297                 7.617012
# 120  0064471101                    The Magician Nephew rack Narnia              8.063437                 7.607201
# 671  088166247X      The Very Best Baby Name Book Whole Wide World              8.002491                 7.604994
# 109  0440406498    The Black Cauldron Chronicles Prydain Paperback              8.093792                 7.602240
# 4    0345361792                                A Prayer Owen Meany              8.071487                 7.597528
# 1    059035342X  Harry Potter Sorcerer Stone Harry Potter Paper...              8.445161                 7.593262
# 11   0553272535                                              Night              8.196129                 7.592551
# 20   0440219078                   The Giver 21st Century Reference              8.153349                 7.590887
# 134  0064409422          The Lion Witch Wardrobe Collector Edition              8.293489                 7.587677
# 22   067168390X                                      Lonesome Dove              8.397790                 7.585583
# 25   0920668372                                   Love You Forever              8.248895                 7.581268
# 9    0307010368                            Snow White Seven Dwarfs              8.227939                 7.578524
# 5    0439064872                Harry Potter Chamber Secrets Book 2              8.084511                 7.576617
# 13   0694003611                          Goodnight Moon Board Book              8.225308                 7.573899
# 18   0440235502                               October Sky A Memoir              8.101679                 7.573248
# 58   0590464639                           The Biggest Pumpkin Ever              8.060757                 7.569270
# 15   0743400526  She Said Yes The Unlikely Martyrdom Cassie Ber...              8.128343                 7.563792
# 0    0440498058                                  A Wrinkle In Time              8.234933                 7.560201
# 305  006440823X                                       Bloomability              8.186722                 7.556408
# 2    0060987561                           I Know This Much Is True              8.011227                 7.554226