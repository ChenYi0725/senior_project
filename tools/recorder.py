def get_dimension(lst):
    # 初始化維度為 1
    dimension = 1
    # 檢查第一個元素是否也是一個列表
    if isinstance(lst[0], list):
        # 遞歸地獲取元素的維度
        dimension = 1 + get_dimension(lst[0])
    return dimension


class recorder:
    def __init__(self):
        self.is_recording = False
        self.recorded_times = 0
        self.needed_times = 21  # 21
        self.is_finish = False
        self.left_feature_per_data = []
        self.rightFeaturePerData = []
        self.original_data = []
        self.location_data = []
        self.accelerate_data = []

    def _fix_location_data(self, feature_per_data):
        if not self.is_recording:
            if feature_per_data:
                del feature_per_data[0]
        return feature_per_data

    def record_one_hand(self, results,feature_per_data):
        if self.recorded_times < self.needed_times:
            feature_per_frame = []
            if results.multi_hand_landmarks:
                # 僅處理第一隻手的數據
                hand_landmarks = results.multi_hand_landmarks[0]

                if hand_landmarks.landmark:
                    for landmark in hand_landmarks.landmark:
                        feature_per_frame.append(landmark.x)
                        feature_per_frame.append(landmark.y)

                feature_per_data.append(feature_per_frame)
                self.recorded_times += 1

        else:
            self.feature_per_data = []
            self.recorded_times = 0
            self.is_recording = False
            self.is_finish = True

        return feature_per_data



    def recordBothHand(self, results, feature_per_data):
        if self.recorded_times < self.needed_times:
            feature_per_frame = []
            left_data_per_frame = []
            right_data_per_frame = []
            if results.multi_hand_landmarks:
                for hand_landmarks, handed in zip(  # 遍歷節點
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # 檢查是否為右手
                    if handed.classification[0].label == "Right":
                        if hand_landmarks.landmark:
                            for landmark in hand_landmarks.landmark:
                                right_data_per_frame.append(landmark.x)
                                right_data_per_frame.append(landmark.y)

                    elif handed.classification[0].label == "Left":
                        if hand_landmarks.landmark:
                            for landmark in hand_landmarks.landmark:
                                left_data_per_frame.append(landmark.x)
                                left_data_per_frame.append(landmark.y)

                feature_per_frame.extend(left_data_per_frame)
                feature_per_frame.extend(right_data_per_frame)
                feature_per_data.append(feature_per_frame)
                self.recorded_times = self.recorded_times + 1

        else:
            self.rightFeaturePerData = []
            self.left_feature_per_data = []
            self.recorded_times = 0
            self.is_recording = False
            self.is_finish = True

        return feature_per_data

    def custom_LR(self,results):
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hands = []
            for hand_landmarks, handed in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                wrist_x = hand_landmarks.landmark[0].x  # 手腕的 X 座標
                hands.append((wrist_x, hand_landmarks, handed))
            hands.sort(key=lambda x: x[0])
            hands[0][2].classification[0].label = "Left"  # 第一隻手（左手）
            hands[1][2].classification[0].label = "Right"  # 第二隻手（右手）
        return results

    def record_2hand_per_frame(self, results):
        feature_per_frame = []
        left_data_per_frame = []
        right_data_per_frame = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handed in zip(  # 遍歷節點
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # 檢查是否為右手
                if handed.classification[0].label == "Right":
                    if hand_landmarks.landmark:
                        for landmark in hand_landmarks.landmark:
                            right_data_per_frame.append(landmark.x)
                            right_data_per_frame.append(landmark.y)

                elif handed.classification[0].label == "Left":
                    if hand_landmarks.landmark:
                        for landmark in hand_landmarks.landmark:
                            left_data_per_frame.append(landmark.x)
                            left_data_per_frame.append(landmark.y)

            feature_per_frame.extend(left_data_per_frame)
            feature_per_frame.extend(right_data_per_frame)
        return feature_per_frame
