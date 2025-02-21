import cv2
import os
import time
import tools.camera

camera = tools.camera.Camera()

# 主資料夾
output_folder = "collected_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 每次啟動建立新的資料夾，以時間命名
session_folder = os.path.join(output_folder, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(session_folder)

image_counter = 0
collecting = False
recording_folder = None  # 當前錄影的資料夾


def collect_images(event, x, y, flags, param):
    global collecting, recording_folder, image_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        collecting = not collecting  # 切換收集狀態
        if collecting:  #如果重新開始新錄影  
            recording_folder = os.path.join(session_folder, time.strftime("%H%M%S"))
            os.makedirs(recording_folder)
            image_counter = 0
            print(f"新資料開始，儲存至資料夾：{recording_folder}")
        else:
            print("錄影暫停")

def hud_om_image(image):
    if collecting:
        cv2.circle(
            image,
            (image.shape[1] - 50, 50),
            min(10, 10),
            (0, 0, 255),
            cv2.FILLED,
        )
    return image

# 監聽滑鼠事件
cv2.namedWindow("Image Collection")
cv2.setMouseCallback("Image Collection", collect_images)

try:
    while True:
        if not camera.camera.isOpened():
            print("Cannot open camera")
            exit()
        is_catchered, BGR_image = camera.get_BGR_image()
    
        BGR_image = hud_om_image(BGR_image)
        cv2.imshow("Image Collection", BGR_image)

        # 如果開始收集，儲存影像
        if collecting and recording_folder is not None:
            image_filename = os.path.join(recording_folder, f"{image_counter}.jpg")
            cv2.imwrite(image_filename, BGR_image)
            print(f"影像已儲存: {image_filename}")
            image_counter += 1

        # 按下 'q' 鍵退出
        if cv2.waitKey(5) == ord("q") or cv2.waitKey(5) == ord("Q"):
            break
        if cv2.getWindowProperty("Image Collection", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    camera.camera.release()
    cv2.destroyAllWindows()
