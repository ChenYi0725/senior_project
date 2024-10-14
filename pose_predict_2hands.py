import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import tools.recorder as rd
import keras
import time
import tensorflow as tf


# from LSTM_2hands_model_trainer import ctcLossFunction


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


def chooseLoadingModel(choose):
    if choose == "lstm_2hand_noCTC_60Features.keras":
        lstmModel = keras.models.load_model(
            "the_precious_working_model/lstm_2hand_noCTC_60Features.keras",
        )
    elif choose == "lstm_2hand_model.h5":
        lstmModel = keras.models.load_model(
            "lstm_2hand_model.h5",
            custom_objects={"ctcLossFunction": ctcLossFunction},
            compile=False,
        )
    return lstmModel


recorder = rd.Recorder()
organizer = do.DataOrganizer()
# def warm_up():
#     organizer.preprocessingData([[[0,0], [0,0]]])
#     print("warm up")
# warm_up()
timeSteps = 21
features = 60

frameReceiver = camera.Camera()
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法

print("loading model")


lstmModel = chooseLoadingModel("lstm_2hand_noCTC_60Features.keras")
print(lstmModel.summary())

print("finish loading")

showResult = "wait"
predictFrequence = 1
predictCount = 0
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# resultsList = [
#     "B'(Back Clockwise)",
#     "B (Back Counter Clockwise)",
#     "D'(Bottom Left)",
#     "D (Bottom Right)",
#     "F (Front Clockwise)",
#     "F' (Front Counter Clockwise)",
#     "L'(Left Down)",
#     "L (Left Up)",
#     "R (Right Down)",
#     "R'(Right Up)",
#     "U (Top Left)",
#     "U'(Top Right)",
#     "Stop",
#     "wait",
# ]
lastResult = 13
resultsList = [
    "B'",
    "B ",
    "D'",
    "D ",
    "F ",
    "F'",
    "L'",
    "L ",
    "R ",
    "R'",
    "U'",
    "U ",
    "Stop",
    "wait",
]
currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 10

# def linearInterpolate(firstTimeStep,secondTimeStep):
#     if (len(firstTimeStep) == len(secondTimeStep)):
#         for i in len(firstTimeStep):

#     return newTimeStep


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, timeSteps, features)
    # 進行預測
    predictData = organizer.preprocessingData(predictData)
    prediction = lstmModel.predict(predictData, verbose=0)  # error
    predictedResult = np.argmax(prediction, axis=1)[0]
    probabilities = prediction[0][predictedResult]
    return predictedResult, probabilities


def blockTheReverseMove(lastCode, currentCode):
    if not (lastCode == 12 and currentCode == 12):
        if (lastCode // 2) == (currentCode // 2):
            return lastCode
        else:
            return currentCode
    return currentCode


def drawResultOnImage(image, resultCode, probabilities):
    global showResult
    global resultsList
    result = resultsList[resultCode]
    showResult = str(result)
    probabilities = str(probabilities)
    cv2.putText(
        image,
        showResult,
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
    cv2.putText(
        image,
        f"timeSteps{len(continuousFeature)}",
        (image.shape[1] - 620, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        f"missCounter{imageHandPosePredict.missCounter}",
        (image.shape[1] - 620, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    return image


def combineAndPredict(currentFeature):
    global continuousFeature
    global predictCount
    global predictFrequence

    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
        # if len(continuousFeature) < 21:
        #     continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        # del continuousFeature[0]
        # continuousFeature.append(currentFeature)
        continuousFeature.append(currentFeature)
        continuousFeature_np = np.array(continuousFeature, dtype="float")
        predictCount = predictCount + 1
        if showResult != "stop":
            if predictCount == predictFrequence:
                predictCount = 0
                predictedResult, probabilities = predict(continuousFeature_np)
                continuousFeature = []
                return predictedResult, probabilities

    return 13, 0


def LeftRightHandClassify(image, results):
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = putTextOnIndexFinger(image, handLandmarks, "L")
            elif handed.classification[0].label == "Right":
                image = putTextOnIndexFinger(image, handLandmarks, "R")
    return image


def isBothExist(results):
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


def imageHandPosePredict(RGBImage):
    global continuousFeature
    global showResult
    global predictCount
    global hands
    global lastResult
    if not hasattr(imageHandPosePredict, "missCounter"):
        imageHandPosePredict.missCounter = 0
    results = hands.process(RGBImage)  # 偵測手掌

    predictedResult = 13
    probabilities = 0
    if isBothExist(results):  # 有雙手
        imageHandPosePredict.missCounter = 0  # miss
        currentFeature = recorder.record2HandPerFrame(results)
        if len(currentFeature) == 84:  # 確認為fearures個特徵
            predictedResult, probabilities = combineAndPredict(currentFeature)
            if probabilities > 0.7:
                if predictedResult < 13 and predictedResult // 2 == lastResult // 2:
                    predictedResult = lastResult  # block reverse move
                else:
                    lastResult = predictedResult
            else:
                predictedResult = lastResult

    else:
        if imageHandPosePredict.missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0
            print("no 2 hands")
        else:
            imageHandPosePredict.missCounter = imageHandPosePredict.missCounter + 1
    return predictedResult, probabilities, results


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

    predictedResult, probabilities, results = imageHandPosePredict(RGBImage)

    BGRImage = drawResultOnImage(
        image=BGRImage,
        resultCode=predictedResult,
        probabilities=probabilities,
    )

    BGRImage = drawNodeOnImage(results=results, image=BGRImage)  # 可移除
    # BGRImage = LeftRightHandClassify(BGRImage, results)  # 可移除

    cv2.imshow("hand tracker", BGRImage)

    if cv2.waitKey(1) == ord("q"):
        break  # 按下 q 鍵停止

    if cv2.getWindowProperty("hand tracker", cv2.WND_PROP_VISIBLE) < 1:
        break


frameReceiver.camera.release()
cv2.destroyAllWindows()
