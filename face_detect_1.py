import cv2

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier('face_recognition_tools/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    return img
def display(frame):
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('src/face_1.jpg')
    frame = cv2.resize(img, (1080, 960))
    img_detected = detect_face(frame)
    display(img_detected)