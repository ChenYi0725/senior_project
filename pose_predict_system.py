import tools.data_organizer as do
import mediapipe as mp
import keras
import tools.recorder as rd
import numpy as np
import time

recorder = rd.recorder()
organizer = do.data_organizer()


recorder = rd.recorder()
organizer = do.data_organizer()
time_steps = 21
# features = 60
features = 36

mp_drawing = mp.solutions.drawing_utils  # 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # 繪圖樣式
mp_hands_solution = mp.solutions.hands  # 偵測手掌方法

lstm_model = keras.models.load_model(
    "lstm_2hand_shirnk_model.keras",
)
show_result = "wait"
predict_frequence = 1
predict_count = 0
hands = mp_hands_solution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


results_list = [
    # "B'",
    # "B ",
    # "D'",
    # "D ",
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
# stopCode = 12
# waitCode = 13
stop_code = 8
wait_code = 9
last_result = wait_code
# currentFeature = []  # 目前畫面的資料
continuous_feature = []  # 目前抓到的前面
miss_counter = 0
max_miss_counter = 10




def is_hand_moving(results, current_feature):  # 檢查finger tips是否被preprocessing 影響
    global continuous_feature
    if not hasattr(is_hand_moving, "last_hand_joints"):
        is_hand_moving.last_hand_joints = []
    if not hasattr(is_hand_moving, "previous_fingertips"):
        is_hand_moving.previous_fingertips = []

    threshold = [0.09, 0.12]
    fingertips_nodes = [8, 4] 
    max_reserve_data = 16
    additional_reserve = 8
    land_mark_adjustment_x = 1
    land_mark_adjustment_y = 1

    left_fingertips = []
    right_fingertips = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                left_wrist = [
                    hand_landmarks.landmark[0].x*land_mark_adjustment_x,
                    hand_landmarks.landmark[0].y*land_mark_adjustment_y,
                ]

                for i in fingertips_nodes:
                    left_fingertips.append(hand_landmarks.landmark[i].x*land_mark_adjustment_x)
                    left_fingertips.append(hand_landmarks.landmark[i].y*land_mark_adjustment_y)
            elif handed.classification[0].label == "Right":

                for i in fingertips_nodes:
                    right_fingertips.append(hand_landmarks.landmark[i].x*land_mark_adjustment_x)
                    right_fingertips.append(hand_landmarks.landmark[i].y*land_mark_adjustment_y)
        current_fingertips = left_fingertips + right_fingertips

        for i in range(len(current_fingertips)):
            current_fingertips[i] = current_fingertips[i] - left_wrist[i % 2]
        current_fingertips = organizer.normalized_one_dimension_list(current_fingertips)
        # --
        is_hand_moving.previous_fingertips.append(current_fingertips)  # 先插入fingertips
        if len(is_hand_moving.previous_fingertips) > max_reserve_data:
            del is_hand_moving.previous_fingertips[0]

        is_hand_moving.previous_fingertips.append(current_fingertips)
        # ------------ 這裡重構
        if len(is_hand_moving.previous_fingertips) > 2:
            current_fingertips = is_hand_moving.previous_fingertips[-1]
            for i in range(
                len(is_hand_moving.previous_fingertips) - 1
            ):  # 不包含最後一個 list
                for j in range(len(current_fingertips)):
                    diff = abs(
                        current_fingertips[j] - is_hand_moving.previous_fingertips[i][j]
                    )
                    if diff > threshold[j % 2]:
                        print(diff)
                        if i > additional_reserve:
                            is_hand_moving.last_hand_joints = is_hand_moving.last_hand_joints[
                                i - additional_reserve :
                            ]
                        return True
        # ----------------
        # 保留時間步
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
                new_time_step = []
                for left_value, right_value in zip(next_time_step, last_time_step):
                    interpolated_value = (
                        (right_value - left_value) * counter / length
                    ) + left_value
                    new_time_step.append(interpolated_value)
                return_list[i + counter - 1] = new_time_step.copy()
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
    predict_data = organizer.preprocess_for_shirnk_model(predict_data)
    try:
        prediction = lstm_model.predict(predict_data, verbose=0)  # error
        predicted_result = np.argmax(prediction, axis=1)[0]
        probabilities = prediction[0][predicted_result]
    except:
        predicted_result = len(results_list) - 1
        probabilities = 0.0

    return predicted_result, probabilities


def block_illegal_result(probabilities, last_result, current_result):
    if probabilities > 0.7:
        if current_result in [stop_code, wait_code]:  # stop, wait 不動
            return current_result

        if current_result == last_result:  # block same move
            return wait_code  # wait

        if last_result != stop_code and (last_result // 2) == (
            current_result // 2
        ):  # block reverse move
            return last_result

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


def combine_and_predict(currentFeature):
    global continuous_feature
    global predict_count
    global predict_frequence

    if len(continuous_feature) < 21:
        continuous_feature.append(currentFeature)
    else:
        del continuous_feature[0]
        continuous_feature.append(currentFeature)
        continuousFeature_np = np.array(continuous_feature, dtype="float")
        predict_count = predict_count + 1
        if show_result != "stop":
            if predict_count == predict_frequence:
                predict_count = 0
                predictedResult, probabilities = predict(continuousFeature_np)
                continuous_feature = []
                is_hand_moving.last_hand_joints = []
                is_hand_moving.previous_fingertips = []
                return predictedResult, probabilities

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
    results = recorder.custom_LR(results)    #修改雙手label
    predicted_result = wait_code
    probabilities = 0

    if is_both_exist(results):  # 如果有雙手
        image_hand_pose_predict.miss_counter = 0
        current_feature = recorder.record_2hand_per_frame(results)
        current_time = time.time()

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
            len(current_feature) == 84 and image_hand_pose_predict.hand_moving_pass_count > 0
        ):  # 確認為特徵的數量
            if (current_time - image_hand_pose_predict.start_time) > 2 and len(
                continuous_feature
            ) > 3:
                continuous_feature = linear_interpolation(
                    continuous_feature
                )  # interpolate to 20 time steps
            predicted_result, probabilities = combine_and_predict(current_feature)
            predicted_result = block_illegal_result(
                probabilities, last_result, predicted_result
            )
            if predicted_result not in [wait_code]:
                result_string = results_list[predicted_result]
                print(result_string)
            image_hand_pose_predict.hand_moving_pass_count = (
                image_hand_pose_predict.hand_moving_pass_count - 1
            )
            #     print(resultsList[predictedResult])
            # # if not predictedResult == 13:
            # #     print(resultsList[predictedResult])

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
    result_string = results_list[predicted_result]
    return result_string, probabilities, results


def clear_current_data():
    global continuous_feature
    continuous_feature = []
    is_hand_moving.previous_fingertips = []
    is_hand_moving.last_hand_joints = []
