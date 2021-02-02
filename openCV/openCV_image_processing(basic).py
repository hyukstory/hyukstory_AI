# < 관심영역 (𝑅𝑒𝑔𝑖𝑜𝑛𝑂𝑓𝐼𝑛𝑡𝑒𝑟𝑒𝑠𝑡,𝑅𝑂𝐼) 지정 >
import cv2
import numpy as np
img = cv2.imread('openCV/ink (2).png')

x = 300; y = 300; w = 100; h = 100  # roi 좌표
roi = img[y : y+h, x : x+w]         # ① roi 지정

print(roi.shape)
cv2.rectangle(roi, (0, 0), (h-1, w-1), (0, 255, 0)) # ② roi 에 사각형 그리기
cv2.imshow("ROI", img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# < 마우스로 관심영역 지정 1. 함수 지정>
import cv2
import numpy as np
isDragging = False                      # 마우스 드래그 상태 저장
x0, y0, w, h = -1, -1, -1, -1           # 영역 선택 좌표 저장
blue, red =  (255, 0, 0), (0, 0, 255)   # 색상 값

def onMouse(event, x, y, flags, param):  # ① 마우스 이벤트 핸들 함수
    global isDragging, x0, y0, img       # 전역변수 참조
    if event == cv2.EVENT_LBUTTONDOWN:   # ② 왼쪽 마우스 버튼 다운 드래그 시작
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:   # ③ 마우스 움직임
        if isDragging:                   # 드래그 진행중
            img_draw = img.copy()        # 사각형 그림 표현을 위한 이미지 복제
            cv2.rectangle(img_draw, (x0, y0), (x,y), blue, 2) # 드레그 진행 영역 표시
            cv2.imshow('mouse drag', img_draw) # 사각형으로 표시된 그림 화면 출력

    elif event == cv2.EVENT_LBUTTONUP:   # ④ 왼쪽 마우스 버튼업
        if isDragging:         # 드래그 중지
            isDragging = False
            w = x - x0          # 드래그 영역 폭 계산
            h = y - y0          # 드래그 영역 높이 계산
            print("x:%d, y:%d, w:%d, h:%d" % (x0,y0,w,h))

            if w > 0 and h > 0:         # 폭과 높이가 음수가 아니면 드래그 방향이 옳음
                img_draw = img.copy()   # 선택 영역에 사각형 그림을 표시할 이미지 복제

            # 선택 영역에 빨간색 사각형 표시
                cv2.rectangle(img_draw,(x0,y0),(x,y), red, 2)
                cv2.imshow('mouse drag', img_draw)  # 빨간색 사각형이 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w]            # 원본 이미지에서 선택 영역만 ROI 로 지정
                cv2.imshow('cropped ROI',roi)          # ROI 지정 영역을 새 창으로 표시
                cv2.moveWindow('cropped ROI', 50, 50)  # 새 창을 화면 좌측 상단으로 이동
                cv2.imwrite('/openCV/cropped_ROI.jpg', roi) # ROI 영역만 파일로 저장
                print('cropped finish')
            else:
                cv2.imshow('mouse drag', img)
                print("좌측 상단에서 우측 하단으로 영역을 드레그 해주세요.")

img = cv2.imread('openCV/ink (2).png')
cv2.imshow('mouse drag', img)
cv2.setMouseCallback('mouse drag', onMouse)  # 마우스 이벤트 등록
cv2.waitKey()
cv2.destroyAllWindows()



# < 마우스로 관심영역 지정 2. selectROI 함수 사용 >
import cv2, numpy as np
img = cv2.imread('openCV/ink (2).png')
x,y,w,h = cv2.selectROI('img', img, True)     # 창의 이름 / ROI 선택을 진행할 이미지 , 선택 영역 중심에 십자 모양표시 여부

if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)                 # ROI 지정 영역을 새 창으로 표시
    cv2.moveWindow('cropped', 50, 50)            # 새 창을 화면 좌측 상단으로 이동
    cv2.imwrite('openCV/cropped2.jpg', roi)    # ROI 영역만 파일로 저장

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(x,y,w,h)



# < 컬러스페이스 변환 (BGR, BGRA) >
import cv2
import numpy as np
img = cv2.imread('openCV/bart_simpson.png')                     # 기본 값 옵션
bgr = cv2.imread('openCV/bart_simpson.png', cv2.IMREAD_COLOR)   #IMREAD_COLOR 옵션
bgra = cv2.imread('openCV/bart_simpson.png', cv2.IMREAD_UNCHANGED)

# 각 옵션에 따른 이미지 SHAPE
print("default: ", img.shape, "color:", bgr.shape, "unchanged:", bgra.shape)

cv2.imshow("bgr", bgr)
cv2.imshow("bgra", bgra)
cv2.imshow("alpha", bgra[:, :, 3])             #알파 채널만 표시
cv2.waitKey(0)
cv2.destroyAllWindows()



# < 그레이 스케일로 변환 >
import cv2
import numpy as np

## 1. 평균값 구하는 알고리즘 직접 구현
img = cv2.imread('openCV/bart_simpson.png')         # 기본 값 옵션
img2 = img.astype(np.uint16)                        # ① dtype 변경
b,g,r = cv2.split(img)                              # ②채널별로 분리
gray1 = ((b + g + r)/ 3 ).astype(np.uint8)          # ③평균 값 연산 후 dtype 변경
### ①에서 dtype 을 uint16 타입으로 변경한 이유는 원래의 dtype 이 uint8 인 경우 평균 값을 구하는 과정에서 3 채널의 값을 합하면
### 255 보다 큰 값으로 나올 수 있으므로 unit16 으로 변경해서 계산을 마치고 다시 코드 ③에서 unit8 로 변경

## 2. cv2.cvtColor() 메소드 사용
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR 을 그레이 스케일로 변경
cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# < 스레시홀딩 (thresholding) >
## 여러 값을 경계점을 기준으로 두 가지 부류로 나누는 것으로 , 바이너리 이미지를 만드는 가장 대표적인 방법

import cv2
import numpy as np
import matplotlib.pylab as plt
# 이미지를 그레이 스케일로 읽기
img = cv2.imread('openCV/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

# 방법① NumPy API 로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img)  # 원본과 동일한 크기의 0 으로 채워진 이미지
thresh_np[ img > 127] = 255     # 127 보다 큰 값만 255 즉, 흰색으로 변경

# 방법② OpenCV API 로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY )
                     # 변환할 이미지 , 경계값 , 경계값 기준에 만족하는 픽셀에 적용할 값, 픽셀 값이 경계값을 넘으면 value 를 지정하고 , 넘지 못하면 0
print(ret)           # 127.0, 바이너리 이미지에 사용된 문턱 값 반환

# 원본과 결과물을 matplotlib 으로 출력
imgs = {'Original': img, 'NumPy API': thresh_np, 'cv2.threshold': thresh_cv}
for i ,  (key, value) in enumerate (imgs.items()) :
    plt.subplot(1, 3, i+ 1)
    plt.title(key)
    plt.imshow(value, cmap = 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()


## < 스레시홀딩 cv2의 다양한 클래스 상수들 >
import cv2
import numpy as np
import matplotlib.pylab as plt
# 이미지를 그레이 스케일로 읽기
img = cv2.imread('openCV/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV )

# 원본과 결과물을 matplotlib 으로 출력
imgs = {'Original': img, 'Binary': binary, 'Binary_INV': binary_inv,
        'Trunc' : trunc, 'Tozero' : tozero, 'Tozero_INV' : tozero_inv}

for i ,  (key, value) in enumerate (imgs.items()) :
    plt.subplot(2, 3, i+ 1)
    plt.title(key)
    plt.imshow(value, cmap = 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()



# < 오츠의 알고리즘 >
## 경계값을 임의로 정해서 픽셀들을 두 부류로 나누고 두 부류의 명암 분포를 반복적으로 구한 다음
## 두 부류의 명암 분포를 가장 균일하게 하는 경계 값을 선택하는 알고리즘

import cv2
import numpy as np
import matplotlib.pylab as plt
# 이미지를 그레이 스케일로 읽기
img = cv2.imread('openCV/receipt.png', cv2.IMREAD_GRAYSCALE)

# 경계 값을 지정하지 않고 OTSU 알고리즘으로 자동 선택
t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 경계값을 -1 로 하면 자동으로 선택
print('otsu threshold:', t)   # Otsu 알고리즘으로 선택된 경계 값 출력

# 원본과 결과물을 matplotlib 으로 출력
imgs = {'Original': img, 'Otsu' : t_otsu}

for i ,  (key, value) in enumerate (imgs.items()) :
    plt.subplot(1, 2, i+ 1)
    plt.title(key)
    plt.imshow(value, cmap = 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()
## 오츠의 알고리즘은 모든 경우의 수에 대해 경계 값을 조사해야 하므로 속도가 느리다는 단점.
## 또한 노이즈가 많은 영상에는 오츠의 알고리즘을 적용해도 좋은 결과를 얻지 못하는 경우가 많다.




# < 적응형 스레시홀드 >
## 이미지를 여러 영역으로 나눈 다음 그 주변 픽셀 값만 가지고 계산을 해서 경계값을 구하는 방법
import cv2
import numpy as np
import matplotlib.pyplot as plt

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수
img = cv2.imread('openCV/receipt.png', cv2.IMREAD_GRAYSCALE) # 그레이 스케일로  읽기

# ---① 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---② 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, blk_size, C)

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1, \
        'Adapted-Mean':th2, 'Adapted-Gaussian': th3}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])


plt.show()


# < 이미지 합치기 >
## 1. 더하기 함수 활용, openCV 함수 활용
import cv2
import numpy as np
import matplotlib.pylab as plt

    # ① 연산에 사용할 이미지 읽기
img1 = cv2.imread('openCV/selfie.png')
img2 = cv2.imread('openCV/gdragon.png')

    # ② 이미지 덧셈
img3 = img1 + img2         ## 더하기 연산 활용
img4 = cv2.add(img1, img2) ## OpenCV 함수

imgs = {'img1':img1, 'img2':img2, 'img1+img2': img3, 'cv.add(img1, img2)': img4}

    # ③ 이미지 출력
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

plt.show()

## 2. 트랙바로 알파 블렌딩 조절하기
import cv2
import numpy as np

win_name = 'Alpha blending'     # 창 이름
trackbar_name = 'fade'          # 트렉바 이름

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    cv2.imshow(win_name, dst)


# ---② 합성 영상 읽기
img1 = cv2.imread('openCV/selfie.png')
img2 = cv2.imread('openCV/gdragon.png')

# ---③ 이미지 표시 및 트렉바 붙이기
cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()




# < 차영상 >
## 두 영상의 차이 파악
## 산업현장에서 도면의 차이를 찾거나 전자제품의 PCB 회로의 오류를 찾는 데도 사용할 수 있고 , 카메라로 촬영한 영상에
## 실시간으로 움직임이 있는지를 알아내는 데도 유용
import numpy as np, cv2

#--① 연산에 필요한 영상을 읽고 그레이스케일로 변환
img1 = cv2.imread('openCV/robot_arm1.jpg')
img2 = cv2.imread('openCV/robot_arm2.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 그레이 스케일로 변환
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 그레이 스케일로 변환

#--② 두 영상의 절대값 차 연산
diff = cv2.absdiff(img1_gray, img2_gray)

#--③ 차 영상을 극대화 하기 위해 쓰레시홀드 처리 및 컬러로 변환
_, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)  # 차이를 극대화 하기위해 1 보다 큰 값은 모두 255 로 바꿈
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)         # 색상을 표현하기 위해 컬러 스케일로 전환
diff_red[:,:,2] = 0

#--④ 두 번째 이미지에 변화 부분 표시
spot = cv2.bitwise_xor(img2, diff_red)  # 원본 이미지는 배경이 흰색이므로 255 를 가지고 있고
                                        # 차영상은 차이가 있는 빨간색 영역을 제외하고는 255 이므로
                                        # XOR 연산을하면 서로 다른 영역인 도면의 그림과 빨간색으로 표시된 차영상 부분이 합성됨.
                                        # (XOR 은 서로 다를 때만 참)

#--⑤ 결과 영상 출력
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('spot', spot)
cv2.waitKey()
cv2.destroyAllWindows()




# < 이미지 합성과 마스킹 >
## 색상에 따라 영역을 떼어내기
'''
dst = cv2.inRange(img, from, to) : 범위에 속하지 않은 픽셀 판단
- img : 입력 영상
- from : 범위의 시작 배열
- to : 범위의 끝 배열
- dst : img 가 from ~ to 에 포함되면 255, 아니면 0 을 픽셀 값으로 하는 배열

HSV : 
- H값 : 색상(빨강: 165 ~180, 0 ~ 15 / 초록: 45 ~ 75 / 파랑: 90 ~120)
- S값 : 채도(색상이 얼마나 순수하게 포함되어 있는지) 0 ~ 255 범위로 표현, 255 는 가장 순수한 색상을 의미
- V값 : 명도(빛이 얼마나 밝은지 어두운지) 0 ~ 255 범위로 표현 , 255 인 경우 가장 밝은 상태
'''

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 신호 이미지 읽어서 HSV로 변환
img = cv2.imread("openCV/lamp.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR을 HSV로 전환

#--② 색상별 영역 지정
blue1 = np.array([90, 50, 50])          # 파랑: 90 ~120
blue2 = np.array([120, 255,255])
green1 = np.array([45, 50,50])          # 초록: 45 ~ 75
green2 = np.array([75, 255,255])
red1 = np.array([0, 50,50])             # 빨강: 165 ~180, 0 ~ 15
red2 = np.array([15, 255,255])
red3 = np.array([165, 50,50])
red4 = np.array([180, 255,255])
yellow1 = np.array([20, 50,50])
yellow2 = np.array([35, 255,255])

# --③ 색상에 따른 마스크 생성
## cv.inRange(img, from, to) 함수는 img에서 from~to 배열 구간에 포함되면 해당 픽셀의 값으로 255를 할당 하고 그렇지 않으면 0을 할당
## 그 결과 이 함수의 반환 결과는 바이너리 스케일이 된다
mask_blue = cv2.inRange(hsv, blue1, blue2)
mask_green = cv2.inRange(hsv, green1, green2)
mask_red = cv2.inRange(hsv, red1, red2)
mask_red2 = cv2.inRange(hsv, red3, red4)
mask_yellow = cv2.inRange(hsv, yellow1, yellow2)

#--④ 색상별 마스크로 색상만 추출
## 위에서 생성된 바이너리 스케일을 mask로 받음
res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
res_green = cv2.bitwise_and(img, img, mask=mask_green)
res_red1 = cv2.bitwise_and(img, img, mask=mask_red)
res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
res_red = cv2.bitwise_or(res_red1, res_red2)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

#--⑤ 결과 출력
imgs = {'original': img, 'blue':res_blue, 'green':res_green,
                            'red':res_red, 'yellow':res_yellow}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(k)
    plt.imshow(v[:,:,::-1])
    plt.xticks([]); plt.yticks([])
plt.show()

