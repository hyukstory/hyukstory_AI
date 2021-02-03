# < 평균 해시 매칭 (average hash matching) >
## 어떤 영상이든 동일한 크기의 하나의 숫자로 변환 되는데, 이때 숫자를 얻기 위해 평균값 을 이용한다
'''
-평균을 얻기 전에 영상을 가로 세로 비율과 무관하게 특정한 크기로 축소 시킨다 .
-픽셀 전체의 평균 값을 구해서 각 픽셀의 값이 평균보다 작으면 0, 크면 1 로 바꾼다.
-0 또는 1 로만 구성된 각 픽셀 값을 1 행 1 열로 변환하는데 , 한 줄로 늘어선 0 과 1 의 숫자들은 한 개의 2 진수 숫자로 볼 수 있습니다
모든영상의 크기를 같은 크기로 축소했기 때문에 모든 2 진수의 비트 개수도 항상 같게 된다.
'''

import cv2

# 영상 읽어서 그레이 스케일로 변환
img = cv2.imread('/openCV/image_set/mouse.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1단계 16x16 크기로 축소 ---①
gray = cv2.resize(gray, (16, 16))

# 2단계 영상의 평균값 구하기 ---②
avg = gray.mean()

# 3단계 평균값을 기준으로 0과 1로 변환 ---③
bin = 1 * (gray > avg)
print(bin)

# 2진수 문자열을 16진수 문자열로 변환 ---④
dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x' % (int(s, 2)))
dhash = ''.join(dhash)
print(dhash)

cv2.namedWindow('mouse', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('mouse', img)
cv2.waitKey(0)

# < 유사도 측정하는 방법 >
## 1. 해밍 거리
# 두 값의 길이가 같아야 계산 가능
# 두 수의 같은 자리의 값이 서로 다른 것이 몇 개인지를 나타내는 거리

import cv2
import numpy as np
import glob

# 영상 읽기 및 표시
img = cv2.imread('/openCV/image_set/mouse.png')
cv2.imshow('query', img)

# 비교할 영상들이 있는 경로 ---①
search_dir = '/openCV/image_set/101_ObjectCategories'


# 이미지를 16x16 크기의 평균 해쉬로 변환 ---②
def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi


# 해밍거리 측정 함수 ---③
def hamming_distance(a, b):
    a = a.reshape(1, -1)  # 1행 1열로 전환
    b = b.reshape(1, -1)  # 1행 1열로 전환
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a != b).sum()
    return distance


# 마우스 영상의 해쉬 구하기 ---④
query_hash = img2hash(img)

# 이미지 데이타 셋 디렉토리의 모든 영상 파일 경로 ---⑤
img_path = glob.glob(search_dir + '/**/*.jpg')
for path in img_path:
    # 데이타 셋 영상 한개 읽어서 표시 ---⑥
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(1)
    # 데이타 셋 영상 한개의 해시  ---⑦
    a_hash = img2hash(img)
    # 해밍 거리 산출 ---⑧
    dst = hamming_distance(query_hash, a_hash)
    if dst / 256 < 0.25:  # 해밍거리 25% 이내만 출력 ---⑨
        print(path, dst / 256)
        cv2.imshow(path, img)

cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()

## 2. 유클리드 거리
# 다음에 다뤄보기


# < 템플릿 매칭 >
## 어떤 물체가 있는 영상을 준비해 두고 그 물체가 포함되어 있을 것이라고 예상할 수 있는 입력과 비교해서
# 물체가 매칭되는 위치를 찾는 것
## 템플릿 영상은 입력 영상보다 그 크기가 항상 작아야 한다.
'''
- result = cv.matchTemplate(img, templ, method[result, mask])
    -img : 입력 영상
    -templ : 템플릿 영상
    -method : 대칭 매서드
        -cv2.TM_SQDIFF_NORMED : 제곱 차이 매칭의 정규화
        -cv2.TM_CCORR : 상관관계 매칭 , 완벽 매칭 : 큰 값 , 나쁜 매칭 : 0
        -cv2.TM_CCORR_NORMED : 상관관계 매칭의 정규화
        -cv2.TM_COEFF : 상관계수 매칭 , 완벽 매칭 : 1, 나쁜 매칭 : 1
        -cv2.TM_CCOEFF_NORMED : 상관계수 매칭의 정규화
    -result : 매칭 결과 . 2 차원 배열
    -mask : TM_SQDIFF, TM_CCORR_NORMED 인 경우 사용할 마스크
    -minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
        -src : 입력 1 차원 배열
        -minVal, maxVal : 배열 전체에서 최소 값 , 최대 값
        -minLoc, maxLoc : 최소 값과 최대 값의 좌표 (x,
'''

import cv2
import numpy as np

img_path = "C:/Users/hyukstory/Desktop/github/hyukstory_AI/openCV/image_set"
# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread(img_path + '/JORDY_family.jpg')  # 기준 이미지 불러오기 (입력 영상)
template = cv2.imread(img_path + '/JORDY.jpg')  # 비교할 이미지 불러오기  (템플릿 이미지)
th, tw = template.shape[:2]  # 비교할 이미지의 행 열 저장

print(img.shape, template.shape)

print(np.shape(template))
print("th : ", th, "tw : ", tw)

cv2.imshow('template', template)  # 비교할 이미지 출력

print("normed_method_1 : ", cv2.matchTemplate(img, template, method=1))  # 정규화된 값 확인


# 3가지 매칭 메서드 순회 - 상관계수 매칭의 정규화, 상관 관계 매칭의 정규화, 제곱 차이 매칭의 정규화
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

for i, method_name in enumerate(methods):  # 정규화를 하나씩 전달
    img_draw = img.copy()  # 원본 이미지 생성
    method = eval(method_name)
    print("method : ", method)  # 5 3 1
    res = cv2.matchTemplate(img, template, method)  # 입력 영상, 템플릿 영상(비교영상), 정규화 방법
    print("res", res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 입력의 1차원을 받는다.
    print(min_val, max_val, min_loc, max_loc)  # 배열 전체에서 최소 값, 최대 값 / 최소 최대 값의 각 좌표

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    print("######", cv2.TM_SQDIFF, "####", cv2.TM_SQDIFF_NORMED)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        print("-------")
        top_left = min_loc  # 정규화한 최소값의 좌표 위치를 반환
        match_val = min_val  # 정규화한 배열 전체의 최소값을 반환
    else:  # 최소값이 좋은 매칭이기에 최대 값을 반환
        top_left = max_loc  # 최대 값의 위치를 반환
        match_val = max_val  # 최대값을 반환
    print("top_left: ", top_left[0], top_left[1])
    print(tw)
    bottom_right = (top_left[0] + tw, top_left[1] + th)  # 이렇게 구한 값을 바탕으로 기존값에서 더해줍니다.
    print(bottom_right)
    cv2.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)  # 이 값을 바탕으로 사각형 그리기
    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left, \
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
