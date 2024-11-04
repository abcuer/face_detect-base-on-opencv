import cv2
from face_detect_1 import detect_face

def detect_face_photo(img):
    img = cv2.imread(img)
    frame = cv2.resize(img, (1080, 960))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier('week_3/face_recognition_tools/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_face_video(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_detected = detect_face(frame)
            cv2.imshow('frame', frame_detected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

#使用摄像头识别
def detect_face_video(num):
    cap = cv2.VideoCapture(num)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_detected = detect_face(frame)
           # 即使识别到人脸也一直保持运行，直到按下q键退出
            cv2.imshow('frame', frame_detected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
def detect_choose():
    print('请选择检测方式：\n1.摄像头\n2.视频\n3.图片')
    print('请输入数字：')
    num = int(input())
    if num == 1:
        detect_face_video(0)
    elif num == 2:
        print('请输入视频路径：')
        video = input()
        detect_face_video(video)    
    elif num == 3:
        print('请输入图片路径：')
        img = input()
        detect_face_photo(img)
    else:
        print('输入有误，请重新输入')


if __name__ == '__main__':
    # detect_face_video('week_3/src/face_detect.mp4')
    detect_choose()