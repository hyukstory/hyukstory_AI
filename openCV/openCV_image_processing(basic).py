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