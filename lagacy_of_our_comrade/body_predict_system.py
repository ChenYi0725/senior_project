import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import tools.recorder as rd
import keras
import time

recorder = rd.recorder()
organizer = do.data_organizer()
timeSteps = 21
features = 60

mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
lstmModel = keras.models.load_model("lagacy_of_our_comrade/body_predict_model.keras")
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
    "R ",
    "R'",
    "U ",
    "U'",
    "Stop",
    "wait",
]
waitCode = 13
stopCode = 12
currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 10


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, timeSteps, features)

    # 進行預測

    predictData = organizer.preprocess_data(predictData)

    prediction = lstmModel.predict(predictData, verbose=0)
    print(prediction[0][12])
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


def isHandMoving(results, currentFeature):  # 檢查finger tips是否被preprocessing 影響
    global continuousFeature
    if not hasattr(isHandMoving, "lastHandJoints"):
        isHandMoving.lastHandJoints = []
    if not hasattr(isHandMoving, "previousFingertips"):
        isHandMoving.previousFingertips = []

    threshold = [0.05, 0.05]
    fingertipsNodes = [0, 8, 4]
    maxReserveData = 5
    additionalReserve = 8
    landMarkAdjustmentX = 1
    landMarkAdjustmentY = 1

    leftFingertips = []
    rightFingertips = []

    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                leftWrist = [
                    handLandmarks.landmark[0].x * landMarkAdjustmentX,
                    handLandmarks.landmark[0].y * landMarkAdjustmentY,
                ]

                for i in fingertipsNodes:
                    leftFingertips.append(
                        handLandmarks.landmark[i].x * landMarkAdjustmentX
                    )
                    leftFingertips.append(
                        handLandmarks.landmark[i].y * landMarkAdjustmentY
                    )
            elif handed.classification[0].label == "Right":

                for i in fingertipsNodes:
                    rightFingertips.append(
                        handLandmarks.landmark[i].x * landMarkAdjustmentX
                    )
                    rightFingertips.append(
                        handLandmarks.landmark[i].y * landMarkAdjustmentY
                    )
        currentFingertips = leftFingertips + rightFingertips

        for i in range(len(currentFingertips)):
            currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
        currentFingertips = organizer.normalized_one_dimension_list(currentFingertips)
        # --
        isHandMoving.previousFingertips.append(currentFingertips)  # 先插入fingertips
        if len(isHandMoving.previousFingertips) > maxReserveData:
            del isHandMoving.previousFingertips[0]

        isHandMoving.previousFingertips.append(currentFingertips)
        # ------------ 這裡重構
        if len(isHandMoving.previousFingertips) > 2:
            currentFingertips = isHandMoving.previousFingertips[-1]
            for i in range(
                len(isHandMoving.previousFingertips) - 1
            ):  # 不包含最後一個 list
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
        # ----------------
        # 保留時間步
        isHandMoving.lastHandJoints.append(currentFeature)
        if len(isHandMoving.lastHandJoints) > maxReserveData:
            del isHandMoving.lastHandJoints[0]

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


def blockIllegalResult(probabilities, lastResult, currentResult):
    if probabilities > 0.7:
        if currentResult in [stopCode, waitCode]:  # stop, wait 不動
            return currentResult

        if currentResult == lastResult:  # block same move
            return waitCode  # wait

        if lastResult != stopCode and (lastResult // 2) == (
            currentResult // 2
        ):  # block reverse move
            return lastResult

        return currentResult
    else:
        return lastResult



def imageHandPosePredict(RGBImage):
    global continuousFeature
    global showResult
    global predictCount
    global hands
    global lastResult
    if not hasattr(imageHandPosePredict, "missCounter"):
        imageHandPosePredict.missCounter = 0  # 用於計算沒有雙手的次數
    if not hasattr(imageHandPosePredict, "handMovingPassCount"):
        imageHandPosePredict.handMovingPassCount = 0  # 用於計算免檢查通行次數
    if not hasattr(imageHandPosePredict, "startTime"):
        imageHandPosePredict.startTime = 0  # 用於計算免檢查通行次數

    results = hands.process(RGBImage)  # 偵測手掌
    results = recorder.custom_LR(results)  # 修改雙手label
    predictedResult = waitCode
    probabilities = 0

    if isBothExist(results):
        imageHandPosePredict.missCounter = 0
        currentFeature = recorder.record_2hand_per_frame(results)
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
            len(currentFeature) == 84 and imageHandPosePredict.handMovingPassCount > 0
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
            if predictedResult not in [waitCode]:
                resultString = resultsList[predictedResult]
                print(resultString)
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
    resultString = resultsList[predictedResult]
    return resultString, probabilities, results
