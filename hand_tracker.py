import cv2
import mediapipe as mp
import tools.camera as camera
import tools.recorder as rd


# 錄製的資料為先左手再右手

mp_drawing = mp.solutions.drawing_utils  # 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # 繪圖樣式
mp_hands_solution = mp.solutions.hands  # 偵測手掌方法
recorder = rd.recorder()
frame_receiver = camera.Camera(0)  # 0->電腦攝影機，1 -> 手機


feature_per_data = []
continuous_feature = []  # 目前抓到的前面
feature_per_process = []  # 這次執行所抓到的資料
current_featute = []  # 目前畫面的資料


def draw_recorded_time(image):
    text = f"Times:{len(feature_per_process)}"
    cv2.putText(
        image,
        text,
        (image.shape[1] - 200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    return image


# def getContinuousFeature(currentFeature):
#     if len(continuousFeature) < 20:
#         continuousFeature.append(currentFeature)
#     else:
#         del continuousFeature[0]
#         continuousFeature.append(currentFeature)
#     return continuousFeature


# def getCurrentFeature(handLandmarks):
#     currentFeature = []
#     if handLandmarks.landmark:
#         for lm in handLandmarks.landmark:
#             currentFeature.append(lm.x)
#             currentFeature.append(lm.y)
#     return currentFeature


def LR_movement(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = put_text_on_index_finger(image, hand_landmarks, "L")
            elif handed.classification[0].label == "Right":
                image = put_text_on_index_finger(image, hand_landmarks, "R")
    return image


def is_LR_exist(results):
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
    else:
        return False
    if is_left and is_right:
        return True
    else:
        return False


def put_text_on_index_finger(image, hand_landmarks, text):
    if hand_landmarks.landmark:
        for lm in hand_landmarks.landmark:
            if (
                lm
                == hand_landmarks.landmark[mp_hands_solution.HandLandmark.INDEX_FINGER_TIP]
            ):
                ih, iw, ic = image.shape
                # print(image.shape)
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


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if recorder.is_recording == False:
            recorder.is_recording = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(feature_per_process) > 0:
            del feature_per_process[-1]


def draw_node_on_image(results, image):  # 將節點和骨架繪製到影像中
    if results.multi_hand_landmarks:
        for hand_marks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_marks,
                mp_hands_solution.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    return image


def recording_sign(image):
    cv2.circle(
        image,
        (image.shape[1] - 50, 50),
        min(10, 10),
        (0, 0, 255),
        cv2.FILLED,
    )

    return image


# mediapipe 啟用偵測手掌
with mp_hands_solution.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    if not frame_receiver.camera.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        is_catchered, BGR_image = frame_receiver.get_BGR_image()
        # BGRImage ->畫面 RGBImage->model

        if not is_catchered:
            print("Cannot receive frame")
            break
        RGB_image = frame_receiver.BGR_to_RGB(BGR_image)
        results = hands.process(RGB_image)  # 偵測手掌
        results = recorder.custom_LR(results)
        # if isLRExist(results):
        #     cv2.putText(
        #         BGRImage,
        #         "Exist",
        #         (300, 300),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (255, 255, 0),
        #         2,
        #     )
        BGR_image = draw_node_on_image(results=results, image=BGR_image)

        if recorder.is_recording:  # and isLRExist(results):
            BGR_image = recording_sign(BGR_image)
            # featurePerData = recorder.recordBothHand(results, featurePerData)
            feature_per_data = recorder.record_one_hand(results, feature_per_data)
            if recorder.is_finish:
                feature_per_process.append(feature_per_data)
                feature_per_data = []
                recorder.is_finish = False
        else:
            pass

        BGR_image = LR_movement(BGR_image, results)
        BGR_image = draw_recorded_time(BGR_image)
        cv2.imshow("hand tracker", BGR_image)
        cv2.setMouseCallback("hand tracker", on_mouse)  # 滑鼠事件

        if cv2.waitKey(5) == ord("q") or cv2.waitKey(5) == ord("Q"):
            break  # 按下 q 鍵停止
        if cv2.getWindowProperty("hand tracker", cv2.WND_PROP_VISIBLE) < 1:
            break

features_string = str(feature_per_process)
# 10 15 10
with open("h.txt", "w") as f:
    features_string = features_string[1:-1]
    f.write(features_string)
print("save in result.txt")
frame_receiver.camera.release()
cv2.destroyAllWindows()
