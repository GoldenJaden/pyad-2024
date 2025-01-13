import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import mae


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""

    pass


def modeling(ratings_df: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.1)
    svd = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.1)
    svd.fit(trainset)

    predictions = svd.test(testset)

    model_mae = mae(predictions)
    print(f'MAE for SVD: {model_mae}')

    if model_mae > 1.3:
        print("Warning: MAE for SVD is above 1.3")
    else:
        print("SVD training successful with MAE below 1.3")

    with open('svd_model.pkl', 'wb') as file:
        pickle.dump(svd, file)
