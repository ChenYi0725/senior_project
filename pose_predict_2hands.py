import cv2
import mediapipe as mp
import numpy as np
import tools.data_oranizer as do
import tools.camera as camera
import tools.recorder as rd
from keras.models import load_model

mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
recorder = rd.Recorder()
frameReceiver = camera.Camera()
organizer = do.DataOrganizer()
lstmModel = load_model("lstm_2hand_model.keras")
resultsList = [
    "Back Clockwise",
    "Back Counter Clockwise",
    "Bottom Left",
    "Bottom Right",
    "Front Clockwise",
    "Front Counter Clockwise",
    "Left Down",
    "Left Up",
    "Right Down",
    "Right Up",
    "Top Left",
    "Top Right",
    "Stop",
]

currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面


def decodedResult(predictedResult):
    decodedResult = resultsList[predictedResult[0]]
    return decodedResult


def predict(continuousFeature, image):
    predictData = np.array([continuousFeature])
    predictData = organizer.getRelativeLocation(predictData)
    prediction = lstmModel.predict(predictData)
    predictedResult = np.argmax(prediction, axis=1)
    image = drawResultOnImage(image=image, result=decodedResult(predictedResult))
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


def combineToContinuous(currentFeature, image):
    global continuousFeature
    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        continuousFeature.append(currentFeature)
        image = predict(continuousFeature, image)
    return image


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
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     recorder.isRecording = True
    pass


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
        # BGRImage -> 畫面， RGBImage -> model
        RGBImage = frameReceiver.BGRToRGB(BGRImage)
        if not isCatchered:
            print("Cannot receive frame")
            break

        results = hands.process(RGBImage)  # 偵測手掌
        if isLRExist(results):
            cv2.putText(
                BGRImage,
                "Exist",
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
        BGRImage = drawNodeOnImage(results=results, image=BGRImage)

        # 加入判斷雙手是否存在
        if isLRExist(results):  # 抓資料
            currentFeature = recorder.record2HandPerFrame(results)
            combineToContinuous(currentFeature, BGRImage)
        else:
            pass

        BGRImage = LRMovement(BGRImage, results)

        cv2.imshow("hand tracker", BGRImage)
        cv2.setMouseCallback("hand tracker", onMouse)  # 滑鼠事件

        if cv2.waitKey(5) == ord("q"):
            break  # 按下 q 鍵停止

frameReceiver.camera.release()
cv2.destroyAllWindows()
