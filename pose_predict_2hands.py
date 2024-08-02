import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import tools.recorder as rd
import keras

mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
recorder = rd.Recorder()
frameReceiver = camera.Camera()
organizer = do.DataOrganizer()
lstmModel = keras.models.load_model("lstm_2hand_model.keras")
showResult = "none"


resultsList = [
    "B'(Back Clockwise)",
    "B (Back Counter Clockwise)",
    "D'(Bottom Left)",
    "D (Bottom Right)",
    "F (Front Clockwise)",
    "F' (Front Counter Clockwise)",
    "L'(Left Down)",
    "L (Left Up)",
    "R (Right Down)",
    "R'(Right Up)",
    "U (Top Left)",
    "U'(Top Right)",
    "Stop",
]

# resultsList = [
#     "B'",
#     "B ",
#     "D'",
#     "D ",
#     "F ",
#     "F'",
#     "L'",
#     "L ",
#     "R ",
#     "R'",
#     "U ",
#     "U'",
#     "Stop",
# ]

currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 5


def decodedResult(predictedResult):
    decodedResult = resultsList[predictedResult]
    return decodedResult


def findResultIndex(result):
    try:
        resultCode = resultsList.index(result)
        return resultCode
    except:
        print("not found, return 12")
        return 12


# def preprocessingContinuousFeature


def predict(continuousFeature, image):
    continuousFeature = np.array(continuousFeature)
    continuousFeature = (continuousFeature - continuousFeature.min()) / (
        continuousFeature.max() - continuousFeature.min()
    )
    # 檢查 continuousFeature 的形狀，應該是 (21, 84)
    if continuousFeature.shape != (21, 84):
        raise ValueError("continuousFeature 的形狀錯誤")

    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, 21, 84)

    # 進行預測
    predictData = organizer.preprocessingData(predictData)
    prediction = lstmModel.predict(predictData)
    predictedResult = np.argmax(prediction, axis=1)[0]  # 確保predictedResult是一個整數

    image = drawResultOnImage(
        image=image,
        resultCode=predictedResult,
        probabilities=prediction[0][predictedResult],
    )

    return image


def blockTheReverseMove(lastCode, currentCode):  # 傳code近來

    if not (lastCode == 12 and currentCode == 12):
        if (lastCode // 2) == (currentCode // 2):

            return lastCode
        else:
            return currentCode
    return currentCode


def drawResultOnImage(image, resultCode, probabilities):
    global showResult

    if probabilities > 0.7:
        lastCode = findResultIndex(showResult)
        print(f"old result {resultCode}")
        resultCode = blockTheReverseMove(lastCode, resultCode)
        print(f"result {resultCode}")
        result = decodedResult(resultCode)

        showResult = str(result)  # result不知道是誰

    probabilities = str(probabilities)
    cv2.putText(
        image,
        probabilities,
        (image.shape[1] - 400, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        showResult,
        (300, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
    return image


def combineAndPredict(currentFeature, image):
    global continuousFeature

    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        continuousFeature.append(currentFeature)

        # 確保 continuousFeature 是一個形狀一致的 NumPy 陣列
        continuousFeature_np = np.array(continuousFeature)

        if continuousFeature_np.shape == (21, len(currentFeature)):
            image = predict(continuousFeature_np, image)
        else:
            print("continuousFeature 形狀錯誤，跳過預測")

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

        # 若雙手同時存在
        if isLRExist(results):  # 抓資料
            missCounter = 0
            currentFeature = recorder.record2HandPerFrame(results)

            combineAndPredict(currentFeature, BGRImage)
        else:  # 若連續沒抓到資料的幀數 > maxMissCounter，則清空先前紀錄的資料
            missCounter = missCounter + 1
            if missCounter > maxMissCounter:
                continuousFeature = []
            pass

        BGRImage = LRMovement(BGRImage, results)

        cv2.imshow("hand tracker", BGRImage)
        cv2.setMouseCallback("hand tracker", onMouse)  # 滑鼠事件

        if cv2.waitKey(5) == ord("q"):
            break  # 按下 q 鍵停止

frameReceiver.camera.release()
cv2.destroyAllWindows()
