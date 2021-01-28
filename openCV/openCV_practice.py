# < 이미지 읽기 >
import cv2
img_file ='C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/ink.png'         # ① 표시할 이미지 경로
img = cv2.imread(img_file)  # ② 이미지를 읽어서 img 변수에 할당

if img is not None:
    cv2.imshow('cat', img)  # ③ 읽은 이미지를 화면에 표시
    cv2.waitKey()           # ④ 키가 입력될 때까지 대기
    cv2.destroyAllWindows()  # ⑤ 아무 키나 누르면 창 모두 닫기
else:
    print("NO IMAGE FILE.")


# < 그레이 스케일로 읽고 저장 >
import cv2
img_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/ink.png'
save_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/ink_gray.png'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # 그레이 스케일로 읽기
cv2.imshow(img_file, img)

cv2.imwrite(save_file, img)      # 파일 저장하는 함수
cv2.waitKey()
cv2.destroyAllWindows()



# < 동영상 및 카메라 프레임 읽기 >
import cv2

video_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/cat.mp4'                 # 동영상 파일 경로
cap = cv2.VideoCapture(video_file)     # 동영상 캡처 객체 생성
if cap.isOpened():                     # 캡처 객체 초기화 확인
    while True:
        ret, img = cap.read()          # 다음 프레임 읽기
        if ret:                        # 프레임 읽기 정상
            cv2.imshow('cat video', img) # 화면에 표시
            cv2.waitKey(25)             # 25ms 지연 (40 fps 로 가정)
        else:                          # 다음 프레임을 읽을 수 없음
            break                      # 재생 완료
else :
    print("can't open video.")         # 캡처 객체 초기화 실패
cap.release()                          # 캡처 자원 반납
cv2.destroyAllWindows()



## < 카메라 (웹캠) 프레임 읽기 >
import cv2

cap = cv2.VideoCapture( 0 )             # ① 0 번 카메라(첫번째) 장치 연결
if cap.isOpened():
    while True:
        ret, img = cap.read()           # 카메라 프레임 읽기
        if ret:
            img = cv2.flip(img, 1)    # 카메라 좌우 반전 (1) / 상하 반전 : 0 / 상하좌우반전 : -1
            cv2.imshow('camera', img)   # 프레임 이미지 표시
            if cv2.waitKey(1) != -1 :   # ② 1ms 동안 키 입력 대기
                break                   # 아무 키나 눌렸으면 중지
        else :
            print("no frame")
            break
else :
    print("can't open camera")
cap.release()
cv2.destroyAllWindows()



'''
• 속성 ID : cv2.CAP_PROP 로 시작하는 상수
- cv2.CAP_PROP_FRAME_WIDTH : 프레임 폭
- cv2.CAP_PROP_FRAME_HEIGHT : 프레임 높이
- cv2.CAP_PROP_FPS : 초당 프레임 수
- cv2.CAP_PROP_POS_MSEC : 동영상 파일의 프레임 위치 (ms)
- cv2.CAP_PROP_AVI_RATIO : 동영상 파일의 상대 위치 (0 : 시작 ,1 : 끝)
- cv2.CAP_PROP_FOURCC : 동영상 파일 코덱 문자
- cv2.CAP_PROP_AUTOFOCUS : 카메라 자동 초점 조절
- cv2.CAP_PROP_ZOOM : 카메라 줌
'''

## < 카메라 비디오 속성 제어 1. FPS>

import cv2
video_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/cat.mp4'   # 동영상 파일 경로

cap = cv2.VideoCapture(video_file)                      # 동영상 캡쳐 객체 생성
if cap.isOpened():                                      # 캡처 객체 초기화 확인
    fps = cap.get(cv2.CAP_PROP_FPS)                     # 동영상의 초당 프레임 수 (fps) 속성 받아오기
    delay = int( 1000 / fps)
    print("FPS : %f, Delay : %dms" % (fps, delay))

    while True :
        ret, img =  cap.read()                             # 다음 프레임 일기
        if ret:                                             # 프레임 읽기 정상
            cv2.imshow("cat video_fps", img)                    # 화면에 표시
            cv2.waitKey(delay)                                  # fps 에 맞게 시간 지연
        else :
            break                                           # 다음 프레임을 읽을 수 없음, 재생 완료
else :
    print("can't open video.")                      # 캡처 객체 초기화 실패
cap.release()                                       # 캡처 자원 반납


## < 카메라 비디오 속성 제어 2. 프레임 폭, 넓이>
import cv2

cap = cv2.VideoCapture(0)                           # 카메라 0 번 장치 연결
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)           # 프레임 폭 값 구하기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)         # 프레임 높이 값 구하기
print("Original width : %d, height : %d" % (width, height))

cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)               # 프레임 폭을 320 으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)              # 프레임 높이를 240 으로 설정
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)           # 재지정한 프레임 폭 값 구하기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)         # 재지정한 프레임 높이 값 구하기
print("Resize width : %d, height : %d" % (width, height))

if cap.isOpened():
    while True :
        ret, img = cap.read()
        if ret:
            img = cv2.flip(img, 1)  # 카메라 좌우 반전 (1) / 상하 반전 : 0 / 상하좌우반전 : -1
            cv2.imshow('camera_frame', img)
            if cv2.waitKey(1) != -1 :
                break

        else :
            print("no frame!")
            break
else :
    print("can't open camera!")
cap.release()
cv2.destroyAllWindows()





## < 비디오 파일 저장 1. 특정 프레임을 이미지로 저장 >

import cv2
cap = cv2.VideoCapture(0)                           # 0 번 카메라 연결
if cap.isOpened():
    while True :
        ret, frame = cap.read()                     # 카메라 프레임 일기
        if ret:
            frame = cv2.flip(frame, 1)  # 카메라 좌우 반전 (1) / 상하 반전 : 0 / 상하좌우반전 : -1
            cv2.imshow('camera_photo', frame)             # 프레임 화면에 표시
            if cv2.waitKey(1) != -1 :               # 아무 키나 누르면
                cv2.imwrite('hyukstory_AI/openCV/photo.jpg', frame)     # 프레임을 'photo.jpg' 에 저장
                break
        else :
            print('no frame')
            break
else:
    print("no camera!")

cap.release()
cv2.destroyAllWindows()


## < 비디오 파일 저장 1. 여러 프레임을 동영상으로 저장 >

import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    file_path = "C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/video_record.mp4"        # 저장할 파일 경로 이름 ①
    fps = 25.40                                                                 # 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*"XVID")               # 인코딩 포맷 문자
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))                     # 프레임 크기
    out = cv2.VideoWriter(file_path, fourcc, fps, size)  # VideoWriter 객체 생성
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # 카메라 좌우 반전 (1) / 상하 반전 : 0 / 상하좌우반전 : -1
            cv2.imshow('camera-recoding', frame)
            out.write(frame)                             # 파일 저장
            if cv2.waitKey(int(1000/fps)) != -1:
                break
        else:
            print("no frame!")
            break
    out.release()
else:
    print("can't open camera")
cap.release()
cv2.destroyAllWindows()



# < 그림 그리기 1. 선 그리기>
'''
cv2.line(img, start, end, color, [thickness, lineType]) : 직선 그리기
- img : 그림 그릴 대상 이미지 , Numpy 배열
- start : 선 시작 지점 좌표 (x,y)
- end : 선 끝 지점 좌표 (x,y)
- color : 선 색상 ,(Blue, Green, Red), 0~255 일반적으로 웹에서 사용하는 RGB 순서와 반대라는 것이 특징
- thickness = 1 : 선 두께
- lineType : 선 그리기 형식
- cv2.LINE_4 : 4 연결 선 알고리즘
- cv2.LINE_8 : 8 연결 선 알고리즘
- cv2.LINE_AA : 안티에일리어싱 계단 현상 없는 선
'''

import cv2
import numpy as np
img = np.full((500, 500, 3), 255, dtype = np.uint8)
cv2.imwrite('C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/paper.jpg', img)


img = cv2.imread('C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/paper.jpg')
cv2.line(img, (50, 50), (150, 50), (255, 0, 0))    # 파란색 1 픽셀 선
cv2.line(img, (200, 50), (300, 50), (0, 255, 0))   # 초록색 1 픽셀 선
cv2.line(img, (350, 50), (450, 50), (0, 0, 5))     # 빨간색 1 픽셀 선

cv2.line(img, (100, 100), (400, 100), (255, 255, 0), 10)  # 하늘색 10 픽셀 선
cv2.line(img, (100, 150), (400, 150), (255, 0, 255), 10)  # 분홍색 10 픽셀 선

cv2.line(img, (100, 350), (400, 400), (0, 0, 255), 20, cv2.LINE_4)   #4 연결선
cv2.line(img, (100, 400), (400, 450), (0, 0, 255), 20, cv2.LINE_8)   #8 연결선
cv2.line(img, (100, 450), (400, 500), (0, 0, 255), 20, cv2.LINE_AA)  # 안티에일리어싱 선

cv2.line(img, (0, 0), (500,500), (0, 0, 255))               # 이미지 전체에 대각선

cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# < 그림 그리기 2. 사각형 그리기>
'''
cv2.rectangle(img, start, end, color, [thickness, lineType]) : 사각형 그리기
- img : 그림 그릴 대상 이미지 , Numpy 배열
- start : 사각형 시작 꼭짓점 (x,y)
- end : 사각형 끝 꼭짓점 (x,y)
- color : 색상 (Blue, Green, Red), 0 ~ 255
- thickness : 선 두께
    - -1 : 채우기 선이 아닌 면을 그리는 것이므로 선의 두께를 지시하는 thickness 에 1 을 지정하면 사각형 면 전체를 color 로
채우기를 합니다
- lineType : 선 타입 , cv2.line() 과 동일
'''

import cv2
img = cv2.imread('C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/paper.jpg')
cv2.rectangle(img, (50,100), (75,50), (255, 0, 0))        # 좌상 우하 좌표로 파란색 사각형 그리기
cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 10)  # 우하 좌상 좌표로 초록색 사각형 그리기
cv2.rectangle(img, (450, 200), (200, 450), (0, 0, 255), -1)  # 우상 좌하 좌표로 빨간색 사각형 채워 그리기
cv2.imshow('rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# < 그림 그리기 3. 원, 타원, 호 그리기 >
'''
• cv2.circle(img, center, radius, color, [thickness, lineType]) : 원 그리기 함수
- img : 그림 대상 이미지
- center : 원점 좌표 (x,Y)
- radius : 원의 반지름
- color : 색상 (Blue, Green, Red)
- thickness : 선 두께 (-1 : 채우기)
- lineType : 선 타입 , cv2. 과 동일

• cv2.ellipse(img, center, axes, angle, from, to, color, [thickness, lineType]) : 호나 타원 그리기 함수
- img : 그림 대상 이미지
- center : 원점 좌표 (x,y)
- axes : 기준 축 길이
- angle : 기준 축 회전 각도
- from, to : 호를 그릴 시작 각도와 끝 각도
'''


import cv2
img = cv2.imread('C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/paper.jpg')

## 원그리기
# 원점 (150, 150),  반지름 100, 파란색
cv2.circle(img,(150, 150), 100, (255, 0, 0))
# 원점 (300, 150), 반지름 70, 초록색, 두께 5
cv2.circle(img,(300, 150), 70, (0, 255, 0), 5)
# 원점 (400, 150), 반지름 50, 빨간색, 채우기
cv2.circle(img,(400, 150), 50, (0, 0, 255), -1)

## 호 그리기
# 원점 (50, 300), 반지름 50, 기준 축 회전 각도 0, 0 도부터 360 도 그리기, 빨간색
cv2.ellipse(img,(50, 300), (50,50), 0, 0, 360, (0, 0, 255))
# 원점 (150, 300), 반지름 50, 기준 축 회전 각도 0, 0 도부터 180 도 그리기, 파란색
cv2.ellipse(img,(150, 300), (50, 50), 0, 0, 180, (255, 0, 0))
# 원점 (200, 300), 반지름 50, 기준 축 회전 각도 0, 181 도부터 360 도 그리기, 초록색
cv2.ellipse(img, (200, 300), (50, 50), 0, 181, 360, (0, 255, 0))

## 타원 그리기
# 원점 (100, 425), 반지름(50, 75), 회전 15 도, 0 도부터 180도 그리기, 빨간색
cv2.ellipse(img, (100, 425), (50, 75), 15, 0, 180, (0, 0, 255))
# 원점 (200, 425), 반지금(100, 50), 회전 45 도, 0 도부터 360도 그리기, 까만색
cv2.ellipse(img, (200, 425), (100, 50), 45, 0, 360, (0, 0, 0))
cv2.imshow('circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()