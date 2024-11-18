import cv2
import time


class Camera:

    def __init__(self,videoDevice):
        self.isCameraOn = True
        self.videoDevice = videoDevice
        self.camera = cv2.VideoCapture(videoDevice)  ##內部攝影機的編號為0
        self.camera.set(cv2.CAP_PROP_FPS, 30)  ##設定攝影機的FPS
        self.imageWidth = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 640
        self.imageHeight = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 480
        self.previousTime = 0

    def _drawFps(self, image):
        fps = 1 / (time.time() - self.previousTime)
        self.previousTime = time.time()
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

    def _updateImageInfo(self, image):
        self.imageHeight = image.shape[0]
        self.imageWidth = image.shape[1]

    def _setImageHud(self, image):
        self._updateImageInfo(image)
        image = self._drawFps(image)
        return image

    def getBGRImage(self):
        isSecreenCaptured, image = self.camera.read()
        image = self.adaptDroidCam(image)
        image = self._setImageHud(image)
        if isSecreenCaptured:
            return isSecreenCaptured, image


    def adaptDroidCam(self, image):
        if self.videoDevice == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


        return image

    def BGRToRGB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
