import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import tools.recorder as rd
import keras
import time
import tensorflow as tf
from keras import regularizers
from keras import layers
import waving_system

# from LSTM_2hands_model_trainer import ctcLossFunction


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


frameReceiver = camera.Camera(0)
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
)


def leftRightHandClassify(image, results):
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = putTextOnIndexFinger(image, handLandmarks, "L")
            elif handed.classification[0].label == "Right":
                image = putTextOnIndexFinger(image, handLandmarks, "R")
    return image


def drawResultOnImage(image, resultString, probabilities):
    global tempResultKeeper
    if resultString == "wait":
        resultString = tempResultKeeper
    else:
        tempResultKeeper = resultString

    probabilities = str(probabilities)
    cv2.putText(
        image,
        resultString,
        (image.shape[1] - 620, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        f"probabilities:{probabilities}",
        (image.shape[1] - 620, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    # cv2.putText(
    #     image,
    #     f"timeSteps{len(where_the_magic_happened.continuousFeature)}",
    #     (image.shape[1] - 620, 200),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    # )
    # cv2.putText(
    #     image,
    #     f"missCounter{where_the_magic_happened.imageHandPosePredict.missCounter}",
    #     (image.shape[1] - 620, 250),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    # )

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


tempResultKeeper = "wait"

# =======================================================

while True:
    if not frameReceiver.camera.isOpened():
        print("Cannot open camera")
        exit()
    isCatchered, BGRImage = frameReceiver.getBGRImage()
    # BGRImage -> 畫面， RGBImage -> model

    RGBImage = frameReceiver.BGRToRGB(BGRImage)

    if not isCatchered:
        print("Cannot receive frame")
        break

    resultString, probabilities, results = waving_system.imageHandPosePredict(RGBImage)
    # resultString, probabilities, results = imageHandPosePredict(RGBImage)
    # noResultImage = BGRImage
    BGRImage = drawResultOnImage(
        image=BGRImage,
        resultString=resultString,
        probabilities=probabilities,
    )
    # results = hands.process(BGRImage)
    # print(waving_system.modeClassification(results))

    BGRImage = drawNodeOnImage(results=results, image=BGRImage)  # 可移除
    BGRImage = leftRightHandClassify(BGRImage, results)  # 可移除

    cv2.imshow("pose predict", BGRImage)

    if cv2.waitKey(1) == ord("q"):
        break  # 按下 q 鍵停止

    if cv2.getWindowProperty("pose predict", cv2.WND_PROP_VISIBLE) < 1:
        break


frameReceiver.camera.release()
cv2.destroyAllWindows()
