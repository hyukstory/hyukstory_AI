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



# < 마우스로 관심영역 지정 >
import cv2
import numpy as np
isDragging = False                 # 마우스 드래그 상태 저장
x0, y0, w, h = -1, -1, -1, -1      # 영역 선택 좌표 저장
blue, red =  (255, 0, 0), (0, 0, 255) # 색상 값

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
                cv2.imshow('red rectangle', img_draw)  # 빨간색 사각형이 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w]            # 원본 이미지에서 선택 영역만 ROI 로 지정
                cv2.imshow('cropped ROI',roi)          # ROI 지정 영역을 새 창으로 표시
                cv2.moveWindow('cropped ROI', 50, 50)  # 새 창을 화면 좌측 상단으로 이동
                cv2.imwrite('/openCV/cropped_ROI.jpg', roi) # ROI 영역만 파일로 저장
                print('cropped finish')
            else:
                cv2.imshow('img', img)
                print("좌측 상단에서 우측 하단으로 영역을 드레그 해주세요.")

img = cv2.imread('openCV/ink (2).png')
cv2.imshow('img drag crop', img)
cv2.setMouseCallback('img drag crop', onMouse)  # 마우스 이벤트 등록
cv2.waitKey()
cv2.destroyAllWindows()




import cv2, numpy as np
img = cv2.imread('openCV/ink (2).png')
x,y,w,h = cv2.selectROI('img', img, False)     # 창의 이름 / ROI 선택을 진행할 이미지 , 선택 영역 중심에 십자 모양표시 여부

if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)                 # ROI 지정 영역을 새 창으로 표시
    cv2.moveWindow('cropped', 0, 0)            # 새 창을 화면 좌측 상단으로 이동
    cv2.imwrite('openCV/cropped2.jpg', roi)    # ROI 영역만 파일로 저장

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(x,y,w,h)