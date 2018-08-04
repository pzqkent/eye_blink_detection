import cv2
import time

cap = cv2.VideoCapture('26122013_230103_cam.wmv')

time.sleep(2)

while(cap.isOpened()):
    ret, frame = cap.read()
    print ret
    print frame
    cv2.imshow('image', frame)
    k = cv2.waitKey(30)

    if (k & 0xff == ord('q')):
        break



cap.release()
cv2.destroyAllWindows()