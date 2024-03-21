import mediapipe


class Recorder:
    def __init__(self):
        self.isRecording = False
        self.recordedTimes = 0
        self.neededTimes = 21  # 31
        self.isFinish = False
        self.originalData = []
        self.locationData = []
        self.accelerateData = []


    def _fixLocationData(self, featurePerData):
        if not self.isRecording:
            if featurePerData:
                del featurePerData[0]
        return featurePerData

    def recordRightData(self, results, featurePerData):
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
                        self.recordedTimes = self.recordedTimes + 1
                        featurePerData.append(featurePerFrame)
                    elif handed.classification[0].label == "Left":
                        pass

        else:
            self.recordedTimes = 0
            self.isRecording = False
            self.isFinish = True
            # print(featurePerData)
        return featurePerData
