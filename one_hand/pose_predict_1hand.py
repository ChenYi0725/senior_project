import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import threading
import tools.recorder as rd
import time
import os
from keras.models import load_model

# 直接儲存2維陣列，ctrl A 直接複製在屁股

featurePerProcess = []
currentFeatute = []
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
organizer = do.DataOrganizer()
lstmModel = load_model("lstm_hand_model.keras")

recorder = rd.Recorder()
frameReceiver = camera.Camera()
featurePerData = []
continuousFeature = []


def drawRecordedTime(image):
    text = f"Times:{len(featurePerProcess)}"
    cv2.putText(
        image,
        text,
        (image.shape[1] - 200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    return image


def drawResultOnImage(image, result):
    result = str(result)
    cv2.putText(
        image,
        result,
        (image.shape[1] - 400, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    return image


def decodeResult(result):
    if result[0] == 0:
        decodedResult = "stop"
    elif result[0] == 1:
        decodedResult = "down"
    else:
        decodedResult = "left"
    return decodedResult


def getContinuousFeature(currentFeature, image):
    global continuousFeature
    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        continuousFeature.append(currentFeature)

        predictData = np.array([continuousFeature])
        predictData = organizer.preprocessingData(predictData)
        prediction = lstmModel.predict(predictData)
        predictedResult = np.argmax(prediction, axis=1)
        image = drawResultOnImage(image=image, result=decodeResult(predictedResult))
    return image


def getCurrentFeature(handLandmarks):
    currentFeature = []
    if handLandmarks.landmark:
        for lm in handLandmarks.landmark:
            currentFeature.append(lm.x)
            currentFeature.append(lm.y)

    return currentFeature


def LRMovement(image, results):
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = putTextOnIndexFinger(image, handLandmarks, "L")
            elif handed.classification[0].label == "Right":
                image = putTextOnIndexFinger(image, handLandmarks, "R")
                image = getContinuousFeature(getCurrentFeature(handLandmarks), image)
    return image


def putTextOnIndexFinger(image, handLandmarks, text):

    if handLandmarks.landmark:
        for lm in handLandmarks.landmark:
            if (
                lm
                == handLandmarks.landmark[mpHandsSolution.HandLandmark.INDEX_FINGER_TIP]
            ):
                ih, iw, ic = image.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.putText(
                    image,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
    return image


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        recorder.isRecording = True


def drawNodeOnImage(results, image):  # 將節點和骨架繪製到影像中
    if results.multi_hand_landmarks:
        for handMarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                image,
                handMarks,
                mpHandsSolution.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style(),
            )
    return image


def drawR(image, x, y):
    boxSize = 110
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + boxSize)
    y2 = int(y + boxSize)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image


def recordingSign(image):
    cv2.circle(
        image,
        (image.shape[1] - 50, 50),
        min(10, 10),
        (0, 0, 255),
        cv2.FILLED,
    )

    return image


# mediapipe 啟用偵測手掌
with mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    if not frameReceiver.camera.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        isCatchered, BGRImage = frameReceiver.getBGRImage()
        # BGRImage ->畫面 RGBImage->model
        RGBImage = frameReceiver.BGRToRGB(BGRImage)

        if not isCatchered:
            print("Cannot receive frame")
            break

        results = hands.process(RGBImage)  # 偵測手掌

        BGRImage = drawNodeOnImage(results=results, image=BGRImage)

        if recorder.isRecording:
            BGRImage = recordingSign(BGRImage)
            featurePerData = recorder.recordRightData(results, featurePerData)
            if recorder.isFinish:
                featurePerProcess.append(featurePerData)
                featurePerData = []
                recorder.isFinish = False
        else:
            pass

        BGRImage = LRMovement(BGRImage, results)  # predict
        BGRImage = drawRecordedTime(BGRImage)
        cv2.imshow("hand tracker", BGRImage)
        cv2.setMouseCallback("hand tracker", onMouse)  # 滑鼠事件
        if cv2.waitKey(5) == ord("q"):
            break  # 按下 q 鍵停止


featuresString = str(featurePerProcess)
with open("test_left.txt", "w") as f:
    f.write(featuresString)
frameReceiver.camera.release()
cv2.destroyAllWindows()
