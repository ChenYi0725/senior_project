import cv2
import time


class Camera:
    def __init__(self, videoDevice):
        self.is_camera_on = True
        self.video_device = videoDevice
        self.camera = cv2.VideoCapture(videoDevice)  ##內部攝影機的編號為0
        self.camera.set(cv2.CAP_PROP_FPS, 30)  ##設定攝影機的FPS
        self.image_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 640
        self.image_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 480
        self.previous_time = 0

    def _draw_fps(self, image):
        fps = 1 / (time.time() - self.previous_time)
        self.previous_time = time.time()
        cv2.putText(
            image,
            f"fps : {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            1,
        )
        return image

    def _update_imageI_info(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

    def _set_image_hud(self, image):
        self._update_imageI_info(image)
        image = self._draw_fps(image)
        return image

    def get_BGR_image(self):
        is_secreen_captured, image = self.camera.read()
        image = cv2.flip(image, 1)
        image = self.adapt_droidCam(image)
        image = self._set_image_hud(image)

        if is_secreen_captured:
            return is_secreen_captured, image

    def adapt_droidCam(self, image):
        if not self.video_device == 0:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image

    def BGR_to_RGB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
