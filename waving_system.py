import tools.data_organizer as do
import mediapipe as mp
import keras
import tools.recorder as rd
import numpy as np
import time
import math

recorder = rd.Recorder()
organizer = do.DataOrganizer()


recorder = rd.Recorder()
organizer = do.DataOrganizer()
timeSteps = 21
# features = 60
features = 42

mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法

lstmModel = keras.models.load_model(
    "exhibit_model.keras",
)
showResult = "wait"
predictFrequence = 1
predictCount = 0
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

checkList = ["H", "v", "stop"]


resultsList = [
    "F'",
    "U'",
    "stop",  # 0
    "F",
    "U",
    "stop",  # 1
    "L",
    "L'",
    "stop",  # 2
    "R",
    "R' ",
    "Stop",  # 3
    "wait",
    "wait",
    "wait",
    "wait",
]

# resultsList = [
#     ["F'", "U'", "stop", "wait"],  # 1
#     ["F ", "U ", "stop", "wait"],  # 2
#     ["L ", "L'", "stop", "wait"],   #3
#     ["R ", "R' ", "Stop", "wait"],  # 4
#     ["wait", "wait", "wait", "wait"],
# ]
# stopCode = 12
# waitCode = 13
stopCode = 3
waitCode = 12
lastResult = waitCode
# currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 10


def calculateDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def isFist(results):
    fingerCount = 0
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):

            wrist = [
                handLandmarks.landmark[0].x,
                handLandmarks.landmark[0].y,
            ]
            for i in [8, 12, 16, 20]:
                finger = [handLandmarks.landmark[i].x, handLandmarks.landmark[i].y]
                palm = [
                    handLandmarks.landmark[i - 3].x,
                    handLandmarks.landmark[i - 3].y,
                ]
                if calculateDistance(finger, wrist) > calculateDistance(palm, wrist):
                    fingerCount = fingerCount + 1
        if fingerCount > 2:
            return False
        else:
            return True


def isLeft(results):
    if results.multi_hand_landmarks:
        if results.multi_handedness[0].classification[0].label == "Left":
            return True
        else:
            return False
    return False


def modeClassification(results):
    if results.multi_hand_landmarks:
        if isFist(results):
            if isLeft(results):
                return 0
            else:
                return 1
        else:
            if isLeft(results):
                return 2
            else:
                return 3
    else:
        return 4


def isHandMoving(results, currentFeature):
    global continuousFeature
    if not hasattr(isHandMoving, "lastHandJoints"):
        isHandMoving.lastHandJoints = []
    if not hasattr(isHandMoving, "previousFingertips"):
        isHandMoving.previousFingertips = []
    if not hasattr(isHandMoving, "lastHand"):
        isHandMoving.lastHand = "none"

    threshold = [0.09, 0.09]
    fingertipsNodes = [8, 4]  # 指尖的節點索引
    maxReserveData = 5  # 最大保留的時間步數
    additionalReserve = 2  # 額外保留數據
    landMarkAdjustmentX = 1
    landMarkAdjustmentY = 1

    currentFingertips = []

    if results.multi_hand_landmarks:
        # if not (results.multi_handedness[0].classification[0].label == isHandMoving.lastHand):
        #     isHandMoving.lastHandJoints = []
        #     isHandMoving.previousFingertips = []
        # if results.multi_handedness[0].classification[0].label == "Left":
        #     isHandMoving.lastHand = "Left"
        # else:
        #     isHandMoving.lastHand = "Right"
        handLandmarks = results.multi_hand_landmarks[0]

        wrist = [
            handLandmarks.landmark[0].x * landMarkAdjustmentX,
            handLandmarks.landmark[0].y * landMarkAdjustmentY,
        ]

        for i in fingertipsNodes:
            currentFingertips.append(handLandmarks.landmark[i].x * landMarkAdjustmentX)
            currentFingertips.append(handLandmarks.landmark[i].y * landMarkAdjustmentY)

        for i in range(len(currentFingertips)):
            currentFingertips[i] = currentFingertips[i] - wrist[i % 2]

        currentFingertips = organizer.normalizedOneDimensionList(currentFingertips)

        isHandMoving.previousFingertips.append(currentFingertips)
        if len(isHandMoving.previousFingertips) > maxReserveData:
            del isHandMoving.previousFingertips[0]

        if len(isHandMoving.previousFingertips) > 2:
            currentFingertips = isHandMoving.previousFingertips[-1]
            for i in range(len(isHandMoving.previousFingertips) - 1):
                for j in range(len(currentFingertips)):
                    diff = abs(
                        currentFingertips[j] - isHandMoving.previousFingertips[i][j]
                    )
                    if diff > threshold[j % 2]:
                        if i > additionalReserve:
                            isHandMoving.lastHandJoints = isHandMoving.lastHandJoints[
                                i - additionalReserve :
                            ]
                        return True

        isHandMoving.lastHandJoints.append(currentFeature)
        if len(isHandMoving.lastHandJoints) > maxReserveData:
            del isHandMoving.lastHandJoints[0]

    return False


# ------------------
def interpolate_number(returnList):
    for i in range(len(returnList)):
        if returnList[i] == None:
            counter = 1
            while (returnList[i + counter]) == None:
                counter = counter + 1
            lastTimeStep = returnList[i + counter]
            nextTimeStep = returnList[i - 1]
            length = counter + 1
            while counter > 0:  # 以time step 為單位
                newTimeStep = []
                for leftValue, rightValue in zip(nextTimeStep, lastTimeStep):
                    interpolated_value = (
                        (rightValue - leftValue) * counter / length
                    ) + leftValue
                    newTimeStep.append(interpolated_value)
                returnList[i + counter - 1] = newTimeStep.copy()
                counter -= 1
    return returnList


def linear_interpolation(targetList):
    global timeSteps
    return_list = [None] * (timeSteps - 1)
    length = len(targetList)
    return_list[0] = targetList[0]  # head and end
    return_list[timeSteps - 2] = targetList[-1]
    for i in range(1, len(targetList) - 1):  # spread the rest of them
        insert_index = ((i * (timeSteps - 2)) // (length - 1)) + 1
        return_list[insert_index] = targetList[i]
    return_list = interpolate_number(return_list)
    return return_list


# -----------------------


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, timeSteps, features)
    # 進行預測
    # predictData = organizer.preprocessingData(predictData)
    predictData = organizer.preprocessExhibitData(predictData)
    try:
        prediction = lstmModel.predict(predictData, verbose=0)  # error
        predictedResult = np.argmax(prediction, axis=1)[0]
        print(f"-----{checkList[predictedResult]}")
        probabilities = prediction[0][predictedResult]
    except:
        predictedResult = len(resultsList) - 1
        probabilities = 0.0

    return predictedResult, probabilities


def blockIllegalResult(probabilities, lastResult, currentResult):
    if probabilities > 0.65:
        if currentResult in [stopCode, waitCode]:  # stop, wait 不動
            return currentResult

        if currentResult == lastResult:  # block same move
            return waitCode  # wait

        # if lastResult != stopCode and (lastResult // 2) == (
        #     currentResult // 2
        # ):  # block reverse move
        #     return lastResult

        return currentResult
    else:
        return lastResult


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


def combineAndPredict(currentFeature):
    global continuousFeature
    global predictCount
    global predictFrequence

    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        continuousFeature.append(currentFeature)
        continuousFeature_np = np.array(continuousFeature, dtype="float")
        predictCount = predictCount + 1
        if showResult != "stop":
            if predictCount == predictFrequence:
                predictCount = 0
                predictedResult, probabilities = predict(continuousFeature_np)
                continuousFeature = []
                isHandMoving.lastHandJoints = []
                isHandMoving.previousFingertips = []
                return predictedResult, probabilities

    return waitCode, 0


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
    if not hasattr(imageHandPosePredict, "startTime"):
        imageHandPosePredict.startTime = 0  # 用於計算免檢查通行次數

    results = hands.process(RGBImage)  # 偵測手掌
    # results = recorder.customLR(results)  # 修改雙手label
    predictedResult = waitCode
    probabilities = 0
    mode = modeClassification(results)

    if results.multi_hand_landmarks:  # 有手
        imageHandPosePredict.missCounter = 0
        currentFeature = recorder.record2HandPerFrame(results)
        currentTime = time.time()

        if imageHandPosePredict.handMovingPassCount == 0:
            if isHandMoving(results, currentFeature):
                imageHandPosePredict.startTime = time.time()
                imageHandPosePredict.handMovingPassCount = timeSteps
                if len(continuousFeature) == 0:
                    continuousFeature.extend(isHandMoving.lastHandJoints[::-1])

            else:
                continuousFeature = []

                pass

        if (
            len(currentFeature) == 42 and imageHandPosePredict.handMovingPassCount > 0
        ):  # 確認為特徵的數量
            # if (currentTime - imageHandPosePredict.startTime) > 2 and len(
            #     continuousFeature
            # ) > 3:
            #     continuousFeature = linear_interpolation(
            #         continuousFeature
            #     )  # interpolate to 20 time steps
            predictedResult, probabilities = combineAndPredict(currentFeature)
            predictedResult = blockIllegalResult(
                probabilities, lastResult, predictedResult
            )
            if predictedResult not in [12, 13, 14, 15]:
                resultString = resultsList[predictedResult + 3 * (mode)]
                print(resultString)
                return resultString, probabilities, results
            imageHandPosePredict.handMovingPassCount = (
                imageHandPosePredict.handMovingPassCount - 1
            )

    else:
        if imageHandPosePredict.missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0
            isHandMoving.lastHandJoints = []
            isHandMoving.previousFingertips = []
            imageHandPosePredict.handMovingPassCount = 0
        else:
            imageHandPosePredict.missCounter += 1

    resultString = resultsList[waitCode]
    return resultString, probabilities, results


def clearCurrentData():
    global continuousFeature
    continuousFeature = []
    isHandMoving.previousFingertips = []
    isHandMoving.lastHandJoints = []
