import mediapipe


def get_dimension(lst):
    # 初始化維度為 1
    dimension = 1
    # 檢查第一個元素是否也是一個列表
    if isinstance(lst[0], list):
        # 遞歸地獲取元素的維度
        dimension = 1 + get_dimension(lst[0])
    return dimension


class Recorder:
    def __init__(self):
        self.isRecording = False
        self.recordedTimes = 0
        self.neededTimes = 11  # 21
        self.isFinish = False
        # self.featurePerData = []
        self.leftFeaturePerData = []
        self.rightFeaturePerData = []
        self.originalData = []
        self.locationData = []
        self.accelerateData = []

    def _fixLocationData(self, featurePerData):
        if not self.isRecording:
            if featurePerData:
                del featurePerData[0]
        return featurePerData

    def recordLeftData(self, results, featurePerData):
        if self.recordedTimes < self.neededTimes:
            featurePerFrame = []
            if results.multi_hand_landmarks:
                for handLandmarks, handed in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # 檢查是否為右手
                    if handed.classification[0].label == "Left":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                featurePerFrame.append(landmark.x)
                                featurePerFrame.append(landmark.y)
                        featurePerData.append(featurePerFrame)
                    elif handed.classification[0].label == "Left":
                        pass

        else:
            self.recordedTimes = 0
            self.isRecording = False
            self.isFinish = True
            # print(featurePerData)
        return featurePerData

    def recordBothHand(self, results, featurePerData):
        if self.recordedTimes < self.neededTimes:
            featurePerFrame = []
            leftDataPerFrame = []
            rightDataPerFrame = []
            if results.multi_hand_landmarks:
                for handLandmarks, handed in zip(  # 遍歷節點
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # 檢查是否為右手
                    if handed.classification[0].label == "Right":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                rightDataPerFrame.append(landmark.x)
                                rightDataPerFrame.append(landmark.y)
                        # self.rightFeaturePerData.append(rightDataPerFrame)
                    elif handed.classification[0].label == "Left":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                leftDataPerFrame.append(landmark.x)
                                leftDataPerFrame.append(landmark.y)
                        # self.leftFeaturePerData.append(leftDataPerFrame)

                featurePerFrame.extend(leftDataPerFrame)
                featurePerFrame.extend(rightDataPerFrame)
                # print(f"check:{featurePerFrame[0]}")
                print(get_dimension(featurePerFrame))
                featurePerData.append(featurePerFrame)
                # print(featurePerFrame)
                self.recordedTimes = self.recordedTimes + 1

        else:
            # featurePerData = self.rightFeaturePerData
            # # featurePerData.append(self.rightFeaturePerData)
            # featurePerData.extend(self.leftFeaturePerData)

            self.rightFeaturePerData = []
            self.leftFeaturePerData = []
            self.recordedTimes = 0
            self.isRecording = False
            self.isFinish = True

        return featurePerData

    def recordRightData(self, results, featurePerData):
        print(len(featurePerData))
        if self.recordedTimes < self.neededTimes:
            featurePerFrame = []
            if results.multi_hand_landmarks:
                for handLandmarks, handed in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # 檢查是否為右手
                    if handed.classification[0].label == "Right":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                featurePerFrame.append(landmark.x)
                                featurePerFrame.append(landmark.y)
                        featurePerData.append(featurePerFrame)
                    elif handed.classification[0].label == "Left":
                        pass

        else:
            self.recordedTimes = 0
            self.isRecording = False
            self.isFinish = True

        return featurePerData
