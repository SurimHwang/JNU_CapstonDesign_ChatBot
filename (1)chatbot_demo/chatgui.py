# tkinter을 사용해 챗봇 GUI 만들기

from tkinter import *
from chatapp import chatbot_response
import gspeech
import time
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="groovy-aquifer-276116-9a00387e56d8.json"


base = Tk()
base.title("전남대 챗봇")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# 채팅창 생성
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# 채팅 스크롤바를 채팅 창에 바인딩하기
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# 메시지 전송 버튼 생성
SendButton = Button(base, font=("Verdana",6,'bold'), text="전송", width="6", height=3,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= lambda: send(message()) )
# 음성 인식 버튼 생성
SpeechButton = Button(base, font=("Verdana",6,'bold'), text="음성인식", width="6", height=3,
                      bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                      command= lambda: send(speech()) )

# 메시지 입력창 생성
EntryBox = Text(base, bd=0, bg="white",width="29", height="3", font="Arial")
#EntryBox.bind("<Return>", send)

def send(msg):
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

def message():
    msg = EntryBox.get("1.0", 'end-1c').strip()  # 입력 창의 텍스트 읽기
    EntryBox.delete("0.0", END)
    return msg

def speech():
    gsp = gspeech.Gspeech()  # 음성->텍스트
    sttmsg = ""
    while True:  # 아래 반복
        stt = gsp.getText()  # 텍스트 가져오기
        # 다음 음성 인식까지 대기
        if stt is None:  # 텍스트 없으면 (음성 없으면)
            return sttmsg
            break  # 종료
        #print(stt)  # 텍스트 출력
        sttmsg += stt
        time.sleep(0.01)  # 0.01msec 주기 반복
        if ('끝내자' in stt):  # 종료 명령어들이 불리면 프로그램 종료
            return sttmsg
            break

# 모든 구성 요소를 화면에 배치
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=60, y=401, height=90)
SpeechButton.place(x=6, y=401, height=90)

base.mainloop()
