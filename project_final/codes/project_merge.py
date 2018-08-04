from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import  face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from imutils.video import FileVideoStream
from sklearn.externals import joblib
from sklearn import preprocessing
import sklearn

x = np.loadtxt("total_input_shuffle.txt", dtype = float, unpack = False)

scaler = preprocessing.StandardScaler().fit(x)

clf = joblib.load('my_model_c50.pkl')

f = open("ear_fortrain2.txt",'a')
f1 = open("ear_after_threshold2.txt",'a')
f2 = open("ear_after_svm2.txt",'a')

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

ear = 0
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True, help = "path to facial landmark predictor")
ap.add_argument("-v", "--video", type = str, default = "", help = "path to input video file")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
COUNTER1 = 0
TOTAL1 = 0
TOTAL2 = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

# vs = FileVideoStream(args["video"]).start()
# fileStream = True

vs = VideoStream(src = 0).start()
#
fileStream = False

time.sleep(1.0)

#===========================#
window = np.zeros([1,13])
frame_index = 0
buffer = list()
#===========================#

while True:

    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width = 720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        leftEye = shape[lStart : lEnd]
        rightEye = shape[rStart : rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1, lineType = cv2.CV_AA)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1, lineType = cv2.CV_AA)

        #============================#
        buffer.append(ear)


        if frame_index >= 12:
            window[0,:] = buffer[frame_index - 12: frame_index + 1]

            if buffer[frame_index - 6] < EYE_AR_THRESH:
                COUNTER1 += 1
                thres_output = 1

            else:
                thres_output = 0

                if COUNTER1 >= 3:
                    TOTAL1 += 1

                COUNTER1 = 0
            f1.write(str(thres_output))
            f1.write('\n')
            # if buffer[frame_index - 6] <= EYE_AR_THRESH:
            #     COUNTER1 += 1
            #     if COUNTER1 >= 4:
            #         TOTAL1 += 1
            #         COUNTER1 = 0



            # window = sklearn.preprocessing.scale(window)

            window = scaler.transform(window)
            print window


            output_predict = clf.predict(window)
            output_predict_write = output_predict[0]
            f2.write(str(output_predict_write))
            f2.write('\n')
            if output_predict == 1:
                COUNTER += 1
            else:
                if COUNTER >= 4:
                    TOTAL2 += 1
                COUNTER = 0

            # COUNTER += output_predict
            # if COUNTER >= EYE_AR_CONSEC_FRAMES:
            #     TOTAL2 += 1
                COUNTER = 0


            TOTAL = round(0.5 * TOTAL1 + 0.5 * TOTAL2)

            # print  buffer

            # TOTAL += output_predict

        # if buffer[frame_index ] < EYE_AR_THRESH:
        #     COUNTER += 1
        #
        # else:
        #
        #     if COUNTER >= 3:
        #         TOTAL += 1
        #
        #     COUNTER = 0

        cv2.putText(frame, "Blinks(SVM): {}".format(TOTAL2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "Blinks(Threshold = {0}): {1}".format(EYE_AR_THRESH,TOTAL1), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "Current EAR: {:.2f}".format(ear), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)

        frame_index += 1
    f.write(str(ear))
    f.write('\n')

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
f.close()
f1.close()
f2.close()





