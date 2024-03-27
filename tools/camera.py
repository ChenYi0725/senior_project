import cv2
import time


class Camera:

    def __init__(self):
        self.isCameraOn = True
        self.camera = cv2.VideoCapture(0)  ##內部攝影機的編號為0
        self.camera.set(cv2.CAP_PROP_FPS, 30)  ##設定攝影機的FPS
        self.previousTime = 0

    def _drawFps(self, image):
        fps = 1 / (time.time() - self.previousTime)
        self.previousTime = time.time()
        cv2.putText(
            image,
            f"fps : {int(fps)}",
            (30, 50),
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
        image = cv2.flip(image, 1)
        self._updateImageInfo(image)
        image = self._drawFps(image)
        return image

    def getBGRImage(self):
        isSecreenCaptured, image = self.camera.read()
        image = self._setImageHud(image)
        if isSecreenCaptured:
            return isSecreenCaptured, image

    def BGRToRGB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
