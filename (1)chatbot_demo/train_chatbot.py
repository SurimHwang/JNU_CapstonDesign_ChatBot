# 해당 파이썬 파일에서 모델을 빌드하고 챗봇을 훈련시키는 스크립트를 작성
# 챗봇에 필요한 패키지를 가져오고 프로젝트에서 사용할 변수를 초기화합니다

'''import nltk
from nltk.stem.lancaster import LancasterStemmer    #입력 문장을 의미 단위로 축소하는 nltk 패키지
                                                    # Lemmatizer: 단어를 기본형으로 전환
stemmer = LancasterStemmer()'''
from konlpy.tag import Kkma          #한글 형태소 분석기
kkma = Kkma()
'''from soylemma import Lemmatizer     #한글 용언 분석기
lemmatizer = Lemmatizer()'''

import json
import pickle

import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from keras.models import Sequential          #Seq2seq 모델
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
#import tflearn
import warnings
warnings.filterwarnings('ignore')

words=[]
classes = []
documents = []
ignore_words = ['?', '!']


# 데이터 파일 가져오기 및 로드
data_file = open('intents.json', 'rt', encoding='UTF8').read()     # 사전 정의된 질의응답이 있는 intents.json 파일
intents = json.loads(data_file)


# 데이터 전처리 단계
# 딥러닝 모델을 만들기 전에 데이터의 다양한 사전 처리를 수행해야 합니다.
for intent in intents['intents']:        # 각 데이터에 대하여 반복
    for pattern in intent['patterns']:       # 각 패턴에 대하여 반복

        # 질의 문장 쪼개기
        '''w = nltk.word_tokenize(pattern)'''     # nltk.word_tokenize()를 이용하여 문장을 단어별로 토큰화. 문장->단어
                                            # 토큰화란 데이터를 문장이나 단어별로 구분하는 것을 뜻함.
        w = kkma.morphs(pattern)  #형태소 분석, 문장을 의미 단위로 쪼갬

        # 단어리스트 생성
        words.extend(w)      # 단어리스트 = 모든 단어 및 어휘

        # 문서리스트 생성
        documents.append((w, intent['tag']))    # 문서 = (단어, 태그)
                                                # 태그란 각 의도를 설명할 만한 한 단어나 문장.
        # 클래스리스트 생성
        if intent['tag'] not in classes:       # 클래스 = 태그 # 중복제거
            classes.append(intent['tag'])

'''# 단어를 기본형으로 줄이기 및 중복 단어 제거'''
# 단어 변환 후 피클 파일을 만들어 예측하는 동안 사용할 파이썬 객체를 저장하는 프로세스입니다.
words = [w.lower() for w in words if w not in ignore_words]       # 단어 리스트의 단어를 소문자로 전환.
                                                                  # '?', '!'는 제외
# 단어, 클래스 리스트 정렬
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# 리스트 출력
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# 단어, 클래스 리스트 객체 파일 생성
pickle.dump(words, open('words.pkl', 'wb'))     # 단어 리스트 객체를 피클 파일로 저장
                                              # 피클 파일이란 데이터가 아니라 파이썬의 객체 자체를 저장하는 파일
pickle.dump(classes, open('classes.pkl', 'wb'))     # 클래스 리스트 객체를 피클 파일로 저장
# 데이터 전처리 단계 끝



# 훈련 데이터 생성 및 챗봇 훈련 단계
# 훈련 데이터 만들기
# 입력은 패턴, 출력은 입력패턴이 속하는 클래스로 이뤄집니다.
training = []
output_empty = [0] * len(classes)       # 출력을 위한 클래스 리스트 길이의 빈 배열을 만듭니다.

# 훈련 세트 만들기, Bag Of Words 이용
for doc in documents:     # 문서 리스트의 각 문서에 대해 반복

    bag = []         # 단어 자루 배열 초기화
    pattern_words = doc[0]     # 패턴에 대해 토큰화된 단어

    # 각 단어를 기본형으로 만들기
    pattern_words = [word.lower() for word in pattern_words]     # 각 단어에 대해 변환

    # 현재 패턴에서 단어 일치가 발견되면 단어 자루에 1을 추가, 발견 안 되면 0을 추가
    for w in words:        # 정렬해 놓았던 단어에 대해
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # 출력은 각 태그에 대해 '0'이고 현재 태그에 대해 '1'입니다. (각 패턴에 대해)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# 특징을 섞고 np.array로 바꿉니다.
random.shuffle(training)
training = np.array(training)

# 훈련과 테스트 리스트를 만듭니다. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
# 훈련 단계 끝


# 모델 구축 단계
# Sequential 모델(순차 모델) 생성 - 3개의 레이어.
# 첫 번째 계층 128 뉴런, 두 번째 계층 64 뉴런, 세 번째 출력 계층에는 여러 개의 뉴런이 포함됩니다.
# keras의 Sequential모델은 인터페이스를 선형으로 연결한다
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # ReLU로 활성화
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # softmax로 활성화, 결과 값이 나올 확률을 알 수 있음, 가장 큰 값으로 분류됨을 확인.
                                                        # 활성화 함수란 입력을 받아 활성, 비활성을 결정하는 데 사용되는 함수
# 컴파일 메서드로 모델 완성
# 최적화 알고리즘 SGD(Stochstic Gradient Descent): 확률적 측면 하강법.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   # Nesterov momentum 알고리즘 사용.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # 모델 학습 평가 기준으로 accuracy 이용

# fit 메서드로 챗봇 훈련 및 저장
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # 200번 학습, 데이터셋 5로 나눠 학습
model.save('chatbot_model.h5', hist)

print("model created")
# 모델 구축 단계 끝