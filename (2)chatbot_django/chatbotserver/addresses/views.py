from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os

#from chatbot.settings import BASE_DIR
from django.http import JsonResponse

import warnings
warnings.filterwarnings('ignore')
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import pickle
import numpy as np
from keras.models import load_model
model = load_model(r'C:\Users\su yeon_Hwang\Desktop\수림 과제\2020 4-1\소프트웨어공학종합프로젝트(캡스톤디자인)\챗봇dj\chatbotserver\addresses\chatbot_model.h5')
#graph = tf.get_default_graph()

import keras.backend as kb
import keras.backend.tensorflow_backend as tb
import json
import random


from konlpy.tag import Kkma      #한글 형태소 분석
kkma = Kkma()

intents = json.loads(open(r'C:\Users\su yeon_Hwang\Desktop\수림 과제\2020 4-1\소프트웨어공학종합프로젝트(캡스톤디자인)\챗봇dj\chatbotserver\addresses\intents.json', 'rt', encoding='UTF-8').read())
# 모델을 훈련시킬 때 만든 피클 파일을 로드
words = pickle.load(open(r'C:\Users\su yeon_Hwang\Desktop\수림 과제\2020 4-1\소프트웨어공학종합프로젝트(캡스톤디자인)\챗봇dj\chatbotserver\addresses\words.pkl', 'rb'))
classes = pickle.load(open(r'C:\Users\su yeon_Hwang\Desktop\수림 과제\2020 4-1\소프트웨어공학종합프로젝트(캡스톤디자인)\챗봇dj\chatbotserver\addresses\classes.pkl', 'rb'))

# 훈련을 예측하려면 훈련하는 동안과 같은 방식으로 입력을 제공해야 합니다.

# 텍스트 전처리 단계
def clean_up_sentence(sentence):

    # 입력 받은 문장을 형태소 단위로 분할
    sentence_words = kkma.morphs(sentence)

    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# 단어 자루 배열 반환 : 문장에 존재하는 단어 자루의 각 단어에 대해 0 또는 1
def bow(sentence, words, show_details=True):

    # 패턴을 분할
    sentence_words = clean_up_sentence(sentence)

    # 단어 자루 - N 단어의 행렬, 어휘 행렬
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:

                # assign 1 if current word is in the vocabulary position
                # 현재 단어가 어휘 위치에 있으면 1을 할당
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):

    # 임계값 미만의 예측을 필터링
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]   # 주어진 입력 데이터로 추론에서 마지막 층의 출력을 예측하여 넘파이배열로 반환
    ERROR_THRESHOLD = 0.25    # 오차 임계값 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # 확률의 강도에 따라 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Create your views here.
@csrf_exempt
# def chat_service(request):
#
#     if request.method == 'POST' :
#         input1 = request.POST['input1']
#         msg = clean_up_sentence(input1)
#         res = chatbot_response(msg)
#         output = dict()
#         output['res'] = res
#         return HttpResponse(json.dumps(output), status=200)
#
#     else:
#         return render(request, 'addresses/chat_test.html')

#질문에 대해 답변 띄우기
def chat_service(request):
    global context
    x = request.POST["msg"]
    resp = chatbot_response(x)
    return HttpResponse(json.dumps({"m":resp}),content_type='application/json')

# def home(request):
#     return render(request,"index.html")

def home(request):
    return render(request,"chat_test.html")
