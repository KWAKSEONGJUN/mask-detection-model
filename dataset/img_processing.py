import cv2
import os


def face_detector(img, cascade, tc, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = cascade.detectMultiScale(gray,  # 입력 이미지
                                       scaleFactor=1.2,  # 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20, 20)  # 탐지 객체 최소 크기
                                       )
    for x, y, w, h in results:
        # 경계 상자 그리기
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        trim_img = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(tc+'{}.jpg'.format(filename)), trim_img)


def img_process(sc, tc, cascade):
    files = os.listdir(sc)

    for filename in files:
        print('Processing Start : {}....'.format(filename), end='')
        img = cv2.imread(sc+filename)
        face_detector(img, cascade, tc, filename)
        print('GOOD!')
    print('TOTAL SUCCESS!!')


xml = '../haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(xml)

sc_path = 'non_processing/with_mask/'
tc_path = 'processing/mask/'
img_process(sc_path, tc_path, cascade)

sc_path = 'non_processing/without_mask/'
tc_path = 'processing/no_mask/'
img_process(sc_path, tc_path, cascade)
