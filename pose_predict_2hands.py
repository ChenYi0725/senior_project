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
import where_the_magic_happened

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

timeSteps = 21
features = 60

frameReceiver = camera.Camera()
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法

# ----
output = 13
inputs = layers.Input(shape=(timeSteps, features), name="input")
lstmLayer = layers.Bidirectional(
    layers.LSTM(
        256,
        activation="tanh",
        kernel_regularizer=regularizers.l2(0.01),
        return_sequences=True,
    )
)(inputs)
lstmLayer = layers.Dense(output + 1, activation="softmax")(lstmLayer)
lstmModel = keras.Model(inputs, lstmLayer)
# lstmModel.load_weights("lstm_2hand_model.keras")
# ----
# lstmModel = chooseLoadingModel("lstm_2hand_noCTC_60Features.keras") #註解以使用純lstm
lstmModel = chooseLoadingModel("lstm_2hand_noCTC_60Features.keras")
showResult = "wait"
predictFrequence = 1
predictCount = 0
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
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
    "R'",
    "R ",
    "U'",
    "U ",
    "Stop",
    "wait",
]

continuousFeature = []  # 目前抓到的全部
missCounter = 0
maxMissCounter = 10


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, timeSteps, features)
    # 進行預測
    predictData = organizer.preprocessingData(predictData)

    prediction = lstmModel.predict(predictData, verbose=0)
    predictedResult = np.argmax(prediction, axis=1)[0]
    probabilities = prediction[0][predictedResult]
    return predictedResult, probabilities

def blockIllegalResult(probabilities, lastResult, currentResult):
    if probabilities > 0.7:
        if currentResult in [12, 13]:  # stop, wait 不動
            return currentResult

        if currentResult == lastResult:  # block same move
            return 13  # wait

        if lastResult != 12 and (lastResult // 2) == (
            currentResult // 2
        ):  # block reverse move
            return lastResult

        return currentResult
    else:
        return lastResult


def drawResultOnImage(image, resultString, probabilities):
    global showResult
    global resultsList
    global continuousFeature
    # result = resultsList[resultCode]
    # showResult = str(result)
    resultString = resultString
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
        f"missCounter{where_the_magic_happened.imageHandPosePredict.missCounter}",
        (image.shape[1] - 620, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    return image


def getIFinger(i):  # just for testing
    fingerName = ["index ", "middle ", "ring "]
    lr = ""
    xy = ""
    if i % 2 == 0:
        xy = "x "
    else:
        xy = "y "
    if i > 5:
        lr = "right "
    else:
        lr = "left "
    finger = fingerName[int((i % 6) / 2)]
    return lr + finger + xy

def interpolate_number(returnList): 
    for i in range(len(returnList)):
        if returnList[i] == None:
            counter = 1
            while (returnList[i+counter]) == None:
                counter = counter + 1
            rightValue = returnList[i+counter]
            leftValue = returnList[i-1]
            length = counter +1
            while counter > 0:
                returnList[i+counter-1] = ((rightValue-leftValue)*counter/length)+leftValue
                counter = counter -1 
    return returnList

def linear_interpolation(targetList):  
    global timeSteps
    return_list = [None] * timeSteps
    length = len(targetList)
    return_list[0] = targetList[0]  # head and end
    return_list[20] = targetList[-1]
    for i in range(1, len(targetList) - 1):  # spread the rest of them
        insert_index = ((i * 19) // (length-1) )+ 1
        return_list[insert_index] = targetList[i]
    return_list = interpolate_number(return_list)
    return return_list


def isHandMoving(results, currentFeature):
    global continuousFeature
    if not hasattr(isHandMoving, "lastHandJoint"):
        isHandMoving.lastHandJoint = []
    if not hasattr(isHandMoving, "lastHandJoint2"):
        isHandMoving.lastHandJoint2 = []
    if not hasattr(isHandMoving, "lastFingertips"):
        isHandMoving.lastFingertips = []
    threshold = [0.02, 0.08]
    fingertipsNodes = [8, 12, 16]  # 4,20

    leftFingertips = []
    rightFingertips = []

    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                leftWrist = [
                    handLandmarks.landmark[0].x,
                    handLandmarks.landmark[0].y,
                ]

                for i in fingertipsNodes:
                    leftFingertips.append(handLandmarks.landmark[i].x)
                    leftFingertips.append(handLandmarks.landmark[i].y)
            elif handed.classification[0].label == "Right":

                for i in fingertipsNodes:
                    rightFingertips.append(handLandmarks.landmark[i].x)
                    rightFingertips.append(handLandmarks.landmark[i].y)
        currentFingertips = leftFingertips + rightFingertips

        for i in range(len(currentFingertips)):
            currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
        currentFingertips = organizer.normalizedOneDimensionList(currentFingertips)

        if isHandMoving.lastFingertips:
            for i in range(len(currentFingertips)):
                fingertipsSAD = abs(
                    currentFingertips[i] - isHandMoving.lastFingertips[i]
                )
                if fingertipsSAD > threshold[i % 2]:  # if moved
                    # print(f"fingertipsSAD:{fingertipsSAD},from:{getIFinger(i)}")
                    isHandMoving.lastFingertips = []
                    isHandMoving.lastHandJoint2 = isHandMoving.lastHandJoint.copy()
                    isHandMoving.lastHandJoint = currentFeature.copy()
                    return True

        isHandMoving.lastFingertips = (
            currentFingertips.copy()
        )  # 把目前的fingertips 保留
        if not isHandMoving.lastHandJoint2:
            isHandMoving.lastHandJoint2 = currentFeature.copy()
        else:
            isHandMoving.lastHandJoint2 = isHandMoving.lastHandJoint.copy()

        isHandMoving.lastHandJoint = currentFeature.copy()

    return False


def combineAndPredict(currentFeature):
    global continuousFeature
    global predictCount
    global predictFrequence

    if len(continuousFeature) < timeSteps:
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

    # 初始化靜態變數
    if not hasattr(imageHandPosePredict, "missCounter"):
        imageHandPosePredict.missCounter = 0  # 用於計算沒有雙手的次數
    if not hasattr(imageHandPosePredict, "handMovingPassCount"):
        imageHandPosePredict.handMovingPassCount = 0  # 用於計算免檢查通行次數

    results = hands.process(RGBImage)  # 偵測手掌

    predictedResult = 13
    probabilities = 0

    if isBothExist(results):  # 如果有雙手
        imageHandPosePredict.missCounter = 0
        currentFeature = recorder.record2HandPerFrame(results)
        print(currentFeature)
        if imageHandPosePredict.handMovingPassCount == 0:
            if isHandMoving(results, currentFeature):  # 如果計數器為0且手動了，開始記錄
                imageHandPosePredict.handMovingPassCount = timeSteps
                if len(continuousFeature) == 0:
                    continuousFeature.append(
                        isHandMoving.lastHandJoint
                    )  # 檢查是否重複append lastFingertips，此處有空可以重構
                    continuousFeature.append(isHandMoving.lastHandJoint2)
            else:
                continuousFeature = []
                pass
        else:
            imageHandPosePredict.handMovingPassCount -= 1

        if (
            len(currentFeature) == 84 and imageHandPosePredict.handMovingPassCount > 0
        ):  # 確認為特徵的數量
            predictedResult, probabilities = combineAndPredict(currentFeature)
            predictedResult = blockIllegalResult(
                probabilities, lastResult, predictedResult
            )
            # if predictedResult not in [12, 13]:
            #     print(resultsList[predictedResult])
            if not predictedResult == 13:
                print(resultsList[predictedResult])

    else:
        if imageHandPosePredict.missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0
            isHandMoving.lastHandJoint = []
            isHandMoving.lastHandJoint2 = []
            imageHandPosePredict.handMovingPassCount = 0
        else:
            imageHandPosePredict.missCounter += 1
    resultString = resultsList[predictedResult]
    return resultString, probabilities, results


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
    
    resultString,probabilities,results = where_the_magic_happened.imageHandPosePredict(RGBImage)
    # resultString, probabilities, results = imageHandPosePredict(RGBImage)

    BGRImage = drawResultOnImage(
        image=BGRImage,
        resultString=resultString,
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
