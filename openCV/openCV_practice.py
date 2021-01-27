# < 이미지 읽기 >
import cv2
img_file ='C:/Users/hyukstory/Desktop/github/hyukstory_AI/ink.png'         # ① 표시할 이미지 경로
img = cv2.imread(img_file)  # ② 이미지를 읽어서 img 변수에 할당

if img is not None:
    cv2.imshow('cat', img)  # ③ 읽은 이미지를 화면에 표시
    cv2.waitKey()           # ④ 키가 입력될 때까지 대기
    cv2.destroyAllWindows()  # ⑤ 아무 키나 누르면 창 모두 닫기
else:
    print("NO IMAGE FILE.")


# < 그레이 스케일로 읽고 저장 >
import cv2
img_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/ink.png'
save_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/ink_gray.png'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # 그레이 스케일로 읽기
cv2.imshow(img_file, img)

cv2.imwrite(save_file, img)      # 파일 저장하는 함수
cv2.waitKey()
cv2.destroyAllWindows()



# < 동영상 및 카메라 프레임 읽기 >
import cv2

video_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/cat.mp4'                 # 동영상 파일 경로
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



## < 카메라 비디오 속성 제어 1. FPS>

import cv2
video_file = 'C:/Users/hyukstory/Desktop/github/hyukstory_AI/cat.mp4'   # 동영상 파일 경로

cap = cv2.VideoCapture(video_file)                  # 동영상 캡쳐 객체 생성
if cap.isOpened():                                      # 캡처 객체 초기화 확인
    fps = cap.get(cv2.CAP_PROP_FPS)                     # 동영상의 초당 프레임 수 (fps) 속성 받아오기
    delay = int( 1000 / fps)
    print("FPS : %f, Delay : %dms" % (fps, delay))

    while True :
        ret, img =  cap.read()                             # 다음 프레임 일기
        if ret:                                             # 프레임 읽기 정상
            cv2.imshow("cat video", img)                        # 화면에 표시
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
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)         #프레임 높이 값 구하기
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
            cv2.imshow('camera', img)
            if cv2.waitKey(1) != -1 :
                break

        else :
            print("no frame!")
            break
else :
    print("can't open camera!")
cap.release()
cv2.destroyAllWindows()




## < 비디오 파일 저장 >

import cv2
cap = cv2.VideoCapture(0)                           # 0 번 카메라 연결
if cap.isOpened():
    while True :
        ret, frame = cap.read()                     # 카메라 프레임 일기
        if ret:
            cv2.imshow('camera', frame)             # 프레임 화면에 표시
            if cv2.waitKey(1) != -1 :               # 아무 키나 누르면
                cv2.imwrite('photo.jpg', frame)     # 프레임을 'photo.jpg' 에 저장
                break
        else :
            print('no frame')
            break
else:
    print("no camera!")

cap.release()
cv2.destroyAllWindows()
