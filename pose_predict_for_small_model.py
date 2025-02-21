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
import pose_predict_system
import where_the_front_magic_happen

# from LSTM_2hands_model_trainer import ctcLossFunction

def LR_movement(image, results):
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                image = put_text_on_index_finger(image, handLandmarks, "L")
            elif handed.classification[0].label == "Right":
                image = put_text_on_index_finger(image, handLandmarks, "R")
    return image


frame_receiver = camera.Camera(1)
mp_drawing = mp.solutions.drawing_utils  # 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # 繪圖樣式
mp_hands_solution = mp.solutions.hands  # 偵測手掌方法


def draw_result_on_image(image, result_string, probabilities):
    global temp_result_keeper
    # textLocationX = image.shape[1] - 620
    text_location_x = 20
    if result_string == "wait":
        result_string = temp_result_keeper
    else:
        temp_result_keeper = result_string

    probabilities = str(probabilities)
    cv2.putText(
        image,
        result_string,
        (text_location_x, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        f"probabilities:{probabilities}",
        (text_location_x, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    # cv2.putText(
    #     image,
    #     f"timeSteps{len(where_the_magic_happened.continuousFeature)}",
    #     (image.shape[1] - 620, 200),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    # )
    # cv2.putText(
    #     image,
    #     f"missCounter{where_the_magic_happened.imageHandPosePredict.missCounter}",
    #     (image.shape[1] - 620, 250),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    # )

    return image


def put_text_on_index_finger(image, hand_landmarks, text):
    if hand_landmarks.landmark:
        for lm in hand_landmarks.landmark:
            if (
                lm
                == hand_landmarks.landmark[mp_hands_solution.HandLandmark.INDEX_FINGER_TIP]
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


temp_result_keeper = "wait"

# =======================================================

while True:
    if not frame_receiver.camera.isOpened():
        print("Cannot open camera")
        exit()
    is_catchered, BGR_image = frame_receiver.get_BGR_image()
    # BGRImage -> 畫面， RGBImage -> model

    RGB_image = frame_receiver.BGR_to_RGB(BGR_image)
    
    if not is_catchered:
        print("Cannot receive frame")
        break

    result_string, probabilities, results = (
        where_the_front_magic_happen.image_hand_pose_predict(RGB_image)
    )

    # resultString, probabilities, results = imageHandPosePredict(RGBImage)
    # noResultImage = BGRImage
    BGR_image = draw_result_on_image(
        image=BGR_image,
        result_string=result_string,
        probabilities=probabilities,
    )

    BGR_image = draw_node_on_image(results=results, image=BGR_image)  # 可移除
    BGR_image = LR_movement(BGR_image, results)  # 可移除

    cv2.imshow("pose predict", BGR_image)

    if cv2.waitKey(1) == ord("q"):
        break  # 按下 q 鍵停止

    if cv2.getWindowProperty("pose predict", cv2.WND_PROP_VISIBLE) < 1:
        break


frame_receiver.camera.release()
cv2.destroyAllWindows()
