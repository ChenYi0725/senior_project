import cv2
import mediapipe as mp
import tools.camera as camera
import tools.recorder as rd
import time
import os


mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
recorder = rd.Recorder()
frameReceiver = camera.Camera()

rightFeaturePerData = []
leftFeaturePerData = []
featurePerData = []
continuousFeature = []  # 目前抓到的前面
featurePerProcess = []  # 這次執行所抓到的資料
currentFeatute = []  # 目前畫面的資料


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


# def getContinuousFeature(currentFeature):
#     if len(continuousFeature) < 20:
#         continuousFeature.append(currentFeature)
#     else:
#         del continuousFeature[0]
#         continuousFeature.append(currentFeature)
#     return continuousFeature


# def getCurrentFeature(handLandmarks):
#     currentFeature = []
#     if handLandmarks.landmark:
#         for lm in handLandmarks.landmark:
#             currentFeature.append(lm.x)
#             currentFeature.append(lm.y)
#     return currentFeature


def LRMovement(image, results):
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = putTextOnIndexFinger(image, handLandmarks, "L")
            elif handed.classification[0].label == "Right":
                image = putTextOnIndexFinger(image, handLandmarks, "R")
    return image


def isLRExist(results):
    isLeft = False
    isRight = False
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                isLeft = True
            elif handed.classification[0].label == "Right":
                isRight = True
    else:
        return False
    if isLeft and isRight:
        return True
    else:
        return False


def putTextOnIndexFinger(image, handLandmarks, text):
    if handLandmarks.landmark:
        for lm in handLandmarks.landmark:
            if (
                lm
                == handLandmarks.landmark[mpHandsSolution.HandLandmark.INDEX_FINGER_TIP]
            ):
                ih, iw, ic = image.shape
                # print(image.shape)
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
        if recorder.isRecording == False:
            recorder.isRecording = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(featurePerProcess) > 0:
            del featurePerProcess[-1]


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
        # if isLRExist(results):
        #     cv2.putText(
        #         BGRImage,
        #         "Exist",
        #         (300, 300),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (255, 255, 0),
        #         2,
        #     )
        BGRImage = drawNodeOnImage(results=results, image=BGRImage)

        if recorder.isRecording and isLRExist(results):
            BGRImage = recordingSign(BGRImage)
            featurePerData = recorder.recordBothHand(results, featurePerData)
            if recorder.isFinish:
                featurePerProcess.append(featurePerData)
                featurePerData = []
                recorder.isFinish = False
        else:
            pass

        # BGRImage = LRMovement(BGRImage, results)
        BGRImage = drawRecordedTime(BGRImage)
        cv2.imshow("hand tracker", BGRImage)
        cv2.setMouseCallback("hand tracker", onMouse)  # 滑鼠事件

        if cv2.waitKey(5) == ord("q") or cv2.waitKey(5) == ord("Q"):
            break  # 按下 q 鍵停止
        if cv2.getWindowProperty("hand tracker", cv2.WND_PROP_VISIBLE) < 1:
            break

featuresString = str(featurePerProcess)
# 10 15 10
with open("result.txt", "w") as f:
    featuresString = featuresString[1:-1]
    f.write(featuresString)

frameReceiver.camera.release()
cv2.destroyAllWindows()
