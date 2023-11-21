import joblib

data = joblib.load(r"C:\Users\USER\hackathon\bookieum_emotion\data.pkl")
collaborative_df = joblib.load(r"C:\Users\USER\hackathon\bookieum_emotion\collaborative_data.pkl")
content_df = joblib.load(r"C:\Users\USER\hackathon\bookieum_emotion\cosine_data.pkl")


# contents based filtering

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action = 'ignore')

from mecab import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# KoNLPy의 Mecab 형태소 분석기 객체 생성
mecab = MeCab()

# TF-IDF 벡터화를 위한 객체 생성 및 변환 완료
book_tfidf = TfidfVectorizer()
book_matrix = book_tfidf.fit_transform(content_df['text'])
# 코사인 유사도 계산
book_similarity = linear_kernel(book_matrix, book_matrix)

# TF-IDF 벡터화를 위한 객체 생성 및 변환 완료
feature_tfidf = TfidfVectorizer()
feature_matrix = feature_tfidf.fit_transform(content_df['features'])
# 코사인 유사도 계산
feature_similarity = cosine_similarity(feature_matrix, feature_matrix)


# 맞춤법 검사
#from hanspell import spell_checker

#def spell_check(sentence):
    #return spell_checker.check(sentence).checked


# 원본 데이터 업데이트 시 변경 될 사항들 업데이트

def update_data_and_cosine_sim(new_data, tfidf):
    global book_matrix, book_similarity, feature_matrix, feature_similarity

    # 원본 데이터 업데이트
    tmp = new_data.copy()

    # Content-based-filtering을 위한 하나의 문자열로 만들기
    tmp['text'] = tmp['Keyword'].apply(lambda x: ' '.join(x))
    tmp['features'] = tmp['genres'].apply(lambda x: ' '.join(x)) + ' ' + tmp['mood'].apply(lambda x: ' '.join(x)) + ' ' + tmp['interest'].apply(lambda x: ' '.join(x))

    df = tmp.copy()
    df = tmp.drop(['categoryName', 'description', 'Keyword', 'genres', 'mood', 'interest'], axis = 1)

    # TF-IDF 벡터화 객체에 새로운 데이터 적용 및 변환 완료
    book_matrix = book_tfidf.fit_transform(df['text'])  # 'content'는 실제 텍스트 컬럼명에 맞게 변경해야 합니다.
    # cosine similarity matrix 업데이트
    book_similarity = cosine_similarity(book_matrix)

    # TF-IDF 벡터화
    feature_matrix = feature_tfidf.fit_transform(df['features'])
    # 코사인 유사도 계산
    feature_similarity = cosine_similarity(feature_matrix, feature_matrix)

    return df

# 사용자가 입력한 문장에 대한 책 추천 10권

def recommend_books_based_on_sentence(sentence, user_read):
    # 사용자가 입력한 문장 맞춤법 검사
    # sentence = spell_checker.check(sentence).checked

    # 사용자가 입력한 문장 (형태소로 분리)
    user_input_sentence = [" ".join(mecab.morphs(sentence))]

    # 이 문장도 TF-IDF 벡터화를 수행합니다.
    user_input_tfidf = book_tfidf.transform(user_input_sentence)

    # 코사인 유사도 계산
    book_similarity_user_input = cosine_similarity(user_input_tfidf, book_matrix)

    # 가장 유사한 책들의 인덱스 찾기
    similar_book_indexes = np.argsort(-book_similarity_user_input.flatten())

    # 사용자가 이미 읽은 책 제외
    similar_book_indexes = [idx for idx in similar_book_indexes if content_df['isbn_id'].iloc[idx] not in user_read]

    # 가장 유사한 10개의 문서 인덱스 가져오기
    top_10_indexes = similar_book_indexes[:10]

    # 원본 데이터프레임에서 해당하는 행 가져오기 (여기서는 'isbn_id' 컬럼만 가져옵니다.)
    similar_books_based_on_user_input = content_df['isbn_id'].iloc[top_10_indexes]

    return similar_books_based_on_user_input.tolist()   # list 형태로 반환


def get_book_titles_by_isbn(isbn_list):
    # isbn_list에 있는 각 isbn에 대응하는 제목 찾기
    book_titles = [content_df[content_df['isbn_id'] == isbn]['title'].values[0] for isbn in isbn_list]

    return book_titles

def recommend_books_based_on_book(isbn, user_read, num_books=10):
    # 입력된 isbn에 해당하는 책의 인덱스 찾기
    idx = content_df[content_df['isbn_id'] == isbn].index[0]

    # 모든 책과의 cosine similarity 값 가져오기
    book_similarity_values = book_similarity[idx]

    # 가장 유사한 책들의 인덱스 찾기
    similar_book_indexes = np.argsort(-book_similarity_values)

    # 사용자가 이미 읽은 책 제외
    similar_book_indexes = [idx for idx in similar_book_indexes if content_df['isbn_id'].iloc[idx] not in user_read]

    # 가장 유사한 num_books개의 문서 인덱스 가져오기
    top_indexes = similar_book_indexes[:num_books]

    # 원본 데이터프레임에서 해당하는 행 가져오기 (여기서는 'isbn_id' 컬럼만 가져옵니다.)
    similar_books_based_on_book = content_df['isbn_id'].iloc[top_indexes]

    return similar_books_based_on_book.tolist()   # list 형태로 반환

# genres, mood, interest로 책 추천

def user_features_recommended_books(genres, mood, interest, user_read):
    # 사용자 features의 입력을 문자열로 합치기
    user_input = ' '.join(genres) + ' ' + ' '.join(mood) + ' ' + ' '.join(interest)

    # 사용자의 입력을 벡터화하기
    user_input_vector = feature_tfidf.transform([user_input])

    # 모든 책에 대한 사용자의 입력과의 유사도를 계산하기
    user_feature_book_similarity = linear_kernel(feature_matrix, user_input_vector)

    # 유사도에 따라 책들을 정렬하기
    sorted_similarity_scores = list(enumerate(user_feature_book_similarity))
    sorted_similarity_scores = sorted(sorted_similarity_scores, key=lambda x: x[1], reverse=True)

    # 사용자가 이미 읽은 책 제외
    sorted_similarity_scores = [score for score in sorted_similarity_scores if content_df['isbn_id'].iloc[score[0]] not in user_read]

    # 가장 유사한 10개의 책의 인덱스를 가져오기
    top10_similar_books = sorted_similarity_scores[0:10]

    # 가장 유사한 10개의 책의 인덱스를 이용하여 책의 'isbn_id'를 반환
    book_indices = [i[0] for i in top10_similar_books]
    return content_df['isbn_id'].iloc[book_indices].tolist()   # list 형태로 반환


# # 사용자의 입력에 따라 책을 추천받기
# genres = ['시대', '전쟁', '과학']
# mood = ['열정', '도전']
# interest = ['영화', '생각', '성공', '심리', '동물', '시간', '사진', '여행', '인간', '시', '그림', '미술']

# print(user_features_recommended_books(genres, mood, interest, user_read))


# collaborative filtering

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

user_book_rating = collaborative_df.pivot_table(index='nickname', columns='isbn_id', values='rating').fillna(0)

# 코사인 유사도 계산
user_similarity = cosine_similarity(user_book_rating)
user_similarity_df = pd.DataFrame(user_similarity, index=user_book_rating.index, columns=user_book_rating.index)

def get_user_recommendations(target_user):
    similar_users = user_similarity_df[target_user]

    # 유사도 순으로 정렬
    similar_users = similar_users.sort_values(ascending=False)

    # 타겟 사용자의 평점을 가져옴
    user1_ratings = user_book_rating.loc[target_user, :]

    # 가장 유사한 사용자의 리스트를 순회
    for user in similar_users.index[1:6]:
        # 각 사용자의 평점을 가져옴
        user2_ratings = user_book_rating.loc[user, :]

        # 타겟 사용자가 평가하지 않은 책 중, 평점이 3 이상인 책만 추천
        recommendations = user2_ratings[(user1_ratings == 0) & (user2_ratings >= 3)]

        # 만약 추천할 책이 있다면 반환
        if not recommendations.empty:
            # Series를 list로 변환, 평점을 제외하고 책의 ID만 반환
            return [item[0] for item in recommendations.sort_values(ascending=False).items()][:10]

    # 모든 사용자를 순회했음에도 추천할 책이 없다면 빈 리스트 반환
    return []

# print(get_user_recommendations('글월마야'))

import face_text
emotion_score=face_text.main()
user_name = '글월마야'
emotion = emotion_score
sentence = '여행가고 싶다. 훌쩍 떠나고 싶다. 해외 여행 가고 싶다.'
user_read = ['9772799628000', '9791198375308']  # 사용자가 읽었던 책 isbn13

# 사용자의 선호장르
genres = ['시대', '전쟁', '과학']
mood = ['열정', '도전']
interest = ['영화', '생각', '성공', '심리', '동물', '시간', '사진', '여행', '인간', '시', '그림', '미술']

# 사용자 리뷰 데이터 5권 전
def recommend_books_under_five_reviews(sentence, user_read, emotion):
    # 문장에 대해 컨텐츠 기반 필터링
    isbn_list_by_sentence = recommend_books_based_on_sentence(sentence, user_read)
    # 선호장르 기반 책 추천
    isbn_list_by_feature = user_features_recommended_books(genres, mood, interest, user_read)

    # 두 리스트를 합치고 중복 제거
    isbn_list = list(set(isbn_list_by_sentence + isbn_list_by_feature))

    # 각 isbn에 대한 감성 점수의 차이 계산
    isbn_with_emotion_diff = [(isbn, abs(data.loc[data['isbn_id'] == isbn, 'emotion_score'].values[0] - emotion)) for isbn in isbn_list]

    # 감성 점수의 차이가 최소인 순서로 정렬
    isbn_with_emotion_diff_sorted = sorted(isbn_with_emotion_diff, key=lambda x: x[1])

    # 차이가 가장 적은 상위 3권의 isbn 반환
    return [isbn for isbn, diff in isbn_with_emotion_diff_sorted[:3]]

print(recommend_books_under_five_reviews(sentence, user_read, emotion))

# 사용지 리뷰 데이터 5권 이상

# 사용자가 읽었던 책 중 가장 평점이 높은 책
user_prefer_isbn = '9791198173898'

def recommend_books_over_five_reviews(sentence, user_name, user_read, user_prefer_isbn, emotion):
    # 문장에 대해 컨텐츠 기반 필터링
    isbn_list_by_sentence = recommend_books_based_on_sentence(sentence, user_read)

    # 사용자 리뷰 데이터 기반 책 추천
    isbn_list_by_review = get_user_recommendations(user_name)

    # isbn_list_by_review이 10권 이하라면,
    # 사용자 선호 도서 기반 책 추천을 통해 10권이 되도록 책 리스트를 추가
    if len(isbn_list_by_review) < 10:
        more_isbn = 10 - len(isbn_list_by_review)
        isbn_list_by_book = recommend_books_based_on_book(user_prefer_isbn, user_read, more_isbn)
    else:
        isbn_list_by_book = []

    # 세 리스트를 합치고 중복 제거
    isbn_list = list(set(isbn_list_by_sentence + isbn_list_by_review + isbn_list_by_book))

    # 각 isbn에 대한 감성 점수의 차이 계산
    isbn_with_emotion_diff = [(isbn, abs(data[data['isbn_id'] == isbn]['emotion_score'].values[0] - emotion)) for isbn in isbn_list]

    # 감성 점수의 차이가 최소인 순서로 정렬
    isbn_with_emotion_diff_sorted = sorted(isbn_with_emotion_diff, key=lambda x: x[1])

    # 차이가 가장 적은 상위 3권의 isbn 반환
    return [isbn for isbn, diff in isbn_with_emotion_diff_sorted[:3]]

print(recommend_books_over_five_reviews(sentence, user_name, user_read, user_prefer_isbn, emotion))
