import cv2
import mediapipe as mp
import numpy as np
import tools.data_organizer as do
import tools.camera as camera
import tools.recorder as rd
import pose_predict_system
import keras
import time



frameReceiver = camera.Camera(0)
#--
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法

#no
def drawResultOnImage(image, resultString, probabilities):
    probabilities = str(probabilities)
    if not (resultString == "wait"):
        pose_predict_system.showResult = resultString
    cv2.putText(
        image,
        probabilities,
        (image.shape[1] - 600, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        pose_predict_system.showResult,
        (image.shape[1] - 600, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    return image


#no
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

    predictedResult, probabilities, results = pose_predict_system.imageHandPosePredict(RGBImage)

    BGRImage = drawResultOnImage(
        image=BGRImage,
        resultString=predictedResult,
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
