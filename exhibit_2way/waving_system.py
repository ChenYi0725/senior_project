import tools.data_organizer as do
import mediapipe as mp
import keras
import tools.recorder as rd
import numpy as np
import time
import math

recorder = rd.recorder()
organizer = do.data_organizer()


recorder = rd.recorder()
organizer = do.data_organizer()
time_steps = 21
# features = 60
features = 42

mp_drawing = mp.solutions.drawing_utils  # 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # 繪圖樣式
mp_hands_solution = mp.solutions.hands  # 偵測手掌方法

lstm_model = keras.models.load_model(
    "exhibit_model.keras",
)
show_result = "wait"
predict_frequence = 1
predict_count = 0
hands = mp_hands_solution.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

check_list = ["H", "v", "stop"]


results_list = [
    "F'",
    "U'",
    "stop",  # 0
    "F ",
    "U ",
    "stop",  # 1
    "L ",
    "L'",
    "stop",  # 2
    "R ",
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
stop_code = 3
wait_code = 12
last_result = wait_code
# currentFeature = []  # 目前畫面的資料
continuous_feature = []  # 目前抓到的前面
miss_counter = 0
max_miss_counter = 10


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def is_fist(results):
    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):

            wrist = [
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
            ]
            for i in [8, 12, 16, 20]:
                finger = [hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]
                palm = [
                    hand_landmarks.landmark[i - 3].x,
                    hand_landmarks.landmark[i - 3].y,
                ]
                if calculate_distance(finger, wrist) > calculate_distance(palm, wrist):
                    finger_count = finger_count + 1
        if finger_count > 2:
            return False
        else:
            return True


def is_left(results):
    if results.multi_hand_landmarks:
        if results.multi_handedness[0].classification[0].label == "Left":
            return True
        else:
            return False
    return False


def mode_classification(results):
    if results.multi_hand_landmarks:
        if is_fist(results):
            if is_left(results):
                return 0
            else:
                return 1
        else:
            if is_left(results):
                return 2
            else:
                return 3
    else:
        return 4


def is_hand_moving(results, current_feature):
    global continuous_feature
    if not hasattr(is_hand_moving, "last_hand_joints"):
        is_hand_moving.last_hand_joints = []
    if not hasattr(is_hand_moving, "previous_fingertips"):
        is_hand_moving.previous_fingertips = []

    threshold = [0.09, 0.09]
    fingertips_nodes = [8, 4]  # 指尖的節點索引
    max_reserve_data = 5  # 最大保留的時間步數
    additional_reserve = 2  # 額外保留數據
    land_mark_adjustment_x = 1
    land_mark_adjustment_y = 1

    current_fingertips = []

    if results.multi_hand_landmarks:
        # if not (results.multi_handedness[0].classification[0].label == isHandMoving.lastHand):
        #     isHandMoving.last_hand_joints = []
        #     isHandMoving.previous_fingertips = []
        # if results.multi_handedness[0].classification[0].label == "Left":
        #     isHandMoving.lastHand = "Left"
        # else:
        #     isHandMoving.lastHand = "Right"
        hand_landmarks = results.multi_hand_landmarks[0]

        wrist = [
            hand_landmarks.landmark[0].x * land_mark_adjustment_x,
            hand_landmarks.landmark[0].y * land_mark_adjustment_y,
        ]

        for i in fingertips_nodes:
            current_fingertips.append(hand_landmarks.landmark[i].x * land_mark_adjustment_x)
            current_fingertips.append(hand_landmarks.landmark[i].y * land_mark_adjustment_y)

        for i in range(len(current_fingertips)):
            current_fingertips[i] = current_fingertips[i] - wrist[i % 2]

        current_fingertips = organizer.normalized_one_dimension_list(current_fingertips)

        is_hand_moving.previous_fingertips.append(current_fingertips)
        if len(is_hand_moving.previous_fingertips) > max_reserve_data:
            del is_hand_moving.previous_fingertips[0]

        if len(is_hand_moving.previous_fingertips) > 2:
            current_fingertips = is_hand_moving.previous_fingertips[-1]
            for i in range(len(is_hand_moving.previous_fingertips) - 1):
                for j in range(len(current_fingertips)):
                    diff = abs(
                        current_fingertips[j] - is_hand_moving.previous_fingertips[i][j]
                    )
                    if diff > threshold[j % 2]:
                        if i > additional_reserve:
                            is_hand_moving.last_hand_joints = is_hand_moving.last_hand_joints[
                                i - additional_reserve :
                            ]
                        return True

        is_hand_moving.last_hand_joints.append(current_feature)
        if len(is_hand_moving.last_hand_joints) > max_reserve_data:
            del is_hand_moving.last_hand_joints[0]

    return False


# ------------------
def interpolate_number(return_list):
    for i in range(len(return_list)):
        if return_list[i] == None:
            counter = 1
            while (return_list[i + counter]) == None:
                counter = counter + 1
            last_time_step = return_list[i + counter]
            next_time_step = return_list[i - 1]
            length = counter + 1
            while counter > 0:  # 以time step 為單位
                newTimeStep = []
                for left_value, right_value in zip(next_time_step, last_time_step):
                    interpolated_value = (
                        (right_value - left_value) * counter / length
                    ) + left_value
                    newTimeStep.append(interpolated_value)
                return_list[i + counter - 1] = newTimeStep.copy()
                counter -= 1
    return return_list


def linear_interpolation(target_list):
    global time_steps
    return_list = [None] * (time_steps - 1)
    length = len(target_list)
    return_list[0] = target_list[0]  # head and end
    return_list[time_steps - 2] = target_list[-1]
    for i in range(1, len(target_list) - 1):  # spread the rest of them
        insert_index = ((i * (time_steps - 2)) // (length - 1)) + 1
        return_list[insert_index] = target_list[i]
    return_list = interpolate_number(return_list)
    return return_list


# -----------------------


def predict(continuous_feature):
    continuous_feature = np.array(continuous_feature)
    predict_data = np.expand_dims(continuous_feature, axis=0)  # (1, timeSteps, features)
    # 進行預測
    # predictData = organizer.preprocessingData(predictData)
    predict_data = organizer.preprocess_exhibit_data(predict_data)
    try:
        prediction = lstm_model.predict(predict_data, verbose=0)  # error
        predicted_result = np.argmax(prediction, axis=1)[0]
        print(f"-----{check_list[predicted_result]}")
        probabilities = prediction[0][predicted_result]
    except:
        predicted_result = len(results_list) - 1
        probabilities = 0.0

    return predicted_result, probabilities


def block_illegal_result(probabilities, last_result, current_result):
    if probabilities > 0.65:
        if current_result in [stop_code, wait_code]:  # stop, wait 不動
            return current_result

        if current_result == last_result:  # block same move
            return wait_code  # wait

        # if lastResult != stopCode and (lastResult // 2) == (
        #     currentResult // 2
        # ):  # block reverse move
        #     return lastResult

        return current_result
    else:
        return last_result


def is_both_exist(results):
    is_left = False
    is_right = False
    if results.multi_hand_landmarks:
        for hand_landmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                is_left = True
            elif handed.classification[0].label == "Right":
                is_right = True

    if is_left and is_right:
        return True
    else:
        return False


def combine_and_predict(current_feature):
    global continuous_feature
    global predict_count
    global predict_frequence

    if len(continuous_feature) < 21:
        continuous_feature.append(current_feature)
    else:
        del continuous_feature[0]
        continuous_feature.append(current_feature)
        continuous_feature_np = np.array(continuous_feature, dtype="float")
        predict_count = predict_count + 1
        if show_result != "stop":
            if predict_count == predict_frequence:
                predict_count = 0
                predicted_result, probabilities = predict(continuous_feature_np)
                continuous_feature = []
                is_hand_moving.last_hand_joints = []
                is_hand_moving.previous_fingertips = []
                return predicted_result, probabilities

    return wait_code, 0


def image_hand_pose_predict(RGB_image):
    global continuous_feature
    global show_result
    global predict_count
    global hands
    global last_result

    # 初始化靜態變數
    if not hasattr(image_hand_pose_predict, "miss_counter"):
        image_hand_pose_predict.miss_counter = 0  # 用於計算沒有雙手的次數
    if not hasattr(image_hand_pose_predict, "hand_moving_pass_count"):
        image_hand_pose_predict.hand_moving_pass_count = 0  # 用於計算免檢查通行次數
    if not hasattr(image_hand_pose_predict, "start_time"):
        image_hand_pose_predict.start_time = 0  # 用於計算免檢查通行次數

    results = hands.process(RGB_image)  # 偵測手掌
    # results = recorder.customLR(results)  # 修改雙手label
    predicted_result = wait_code
    probabilities = 0
    mode = mode_classification(results)

    if results.multi_hand_landmarks:  # 有手
        image_hand_pose_predict.miss_counter = 0
        current_feature = recorder.record_2hand_per_frame(results)
        currentTime = time.time()

        if image_hand_pose_predict.hand_moving_pass_count == 0:
            if is_hand_moving(results, current_feature):
                image_hand_pose_predict.start_time = time.time()
                image_hand_pose_predict.hand_moving_pass_count = time_steps
                if len(continuous_feature) == 0:
                    continuous_feature.extend(is_hand_moving.last_hand_joints[::-1])

            else:
                continuous_feature = []

                pass

        if (
            len(current_feature) == 42 and image_hand_pose_predict.hand_moving_pass_count > 0
        ):  # 確認為特徵的數量
            # if (currentTime - imageHandPosePredict.start_time) > 2 and len(
            #     continuous_Feature
            # ) > 3:
            #     continuousFeature = linear_interpolation(
            #         continuous_feature
            #     )  # interpolate to 20 time steps
            predicted_result, probabilities = combine_and_predict(current_feature)
            predicted_result = block_illegal_result(
                probabilities, last_result, predicted_result
            )
            if predicted_result not in [12, 13, 14, 15]:
                result_string = results_list[predicted_result + 3 * (mode)]
                print(result_string)
                return result_string, probabilities, results
            image_hand_pose_predict.hand_moving_pass_count = (
                image_hand_pose_predict.hand_moving_pass_count - 1
            )

    else:
        if image_hand_pose_predict.miss_counter >= max_miss_counter:
            continuous_feature = []
            show_result = "wait"
            predict_count = 0
            is_hand_moving.last_hand_joints = []
            is_hand_moving.previous_fingertips = []
            image_hand_pose_predict.hand_moving_pass_count = 0
        else:
            image_hand_pose_predict.miss_counter += 1

    result_string = results_list[wait_code]
    return result_string, probabilities, results


def clearCurrentData():
    global continuous_feature
    continuous_feature = []
    is_hand_moving.previous_fingertips = []
    is_hand_moving.last_hand_joints = []
