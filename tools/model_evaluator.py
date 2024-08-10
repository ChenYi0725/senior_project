import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import data_organizer as do
from tools import data_organizer as do


class ModelEvaluator(Callback):
    def __init__(self, label):
        super(ModelEvaluator, self).__init__()
        self.organizer = do.DataOrganizer()
        self.losses = []
        self.label = label
        self.dataLengthList = []
        self.testData, self.testlabel = self.createTestData()
        print(self.testData.shape)
        print(self.testlabel.shape)

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))

    def on_train_end(self, logs=None):
        self.drawLossFunction()
        # self.drawConfusionMatrix()

    def drawLossFunction(self, logs={}):
        # 繪製損失值圖表
        plt.plot(self.losses)
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def drawConfusionMatrix(self):
        yPred = self.model.predict(self.testData)
        yPredClasses = np.argmax(yPred, axis=1)
        yTrue = np.argmax(self.testlabel, axis=1)

        conf_matrix = confusion_matrix(yTrue, yPredClasses)

        # 繪製混淆矩陣
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=self.label,
            yticklabels=self.label,
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # ----------------------------------------------------
    def initData(
        self, inputList
    ):  # inputList.shape = (data numbers, time step, features)
        inputList = np.array(inputList)
        self.dataLengthList.append(len(inputList))
        inputList = self.organizer.preprocessingData(inputList)
        return inputList

    def createTestData(self):
        bp = self.organizer.getDataFromTxt("test_data_set/b'_test")
        b = self.organizer.getDataFromTxt("test_data_set/b_test")
        dp = self.organizer.getDataFromTxt("test_data_set/d'_test")
        d = self.organizer.getDataFromTxt("test_data_set/d_test")
        f = self.organizer.getDataFromTxt("test_data_set/f_test")
        fp = self.organizer.getDataFromTxt("test_data_set/f'_test")
        lp = self.organizer.getDataFromTxt("test_data_set/l'_test")
        l = self.organizer.getDataFromTxt("test_data_set/l_test")
        r = self.organizer.getDataFromTxt("test_data_set/r_test")
        rp = self.organizer.getDataFromTxt("test_data_set/r'_test")
        u = self.organizer.getDataFromTxt("test_data_set/u_test")
        up = self.organizer.getDataFromTxt("test_data_set/u'_test")
        stop = self.organizer.getDataFromTxt("test_data_set/stop_test")
        # ---
        bp = self.initData(bp)
        b = self.initData(b)
        dp = self.initData(dp)
        d = self.initData(d)
        f = self.initData(f)
        fp = self.initData(fp)
        lp = self.initData(lp)
        l = self.initData(l)
        r = self.initData(r)
        rp = self.initData(rp)
        u = self.initData(u)
        up = self.initData(up)
        stop = self.initData(stop)
        # ------
        testData = np.concatenate(
            (
                bp,
                b,
                dp,
                d,
                f,
                fp,
                lp,
                l,
                r,
                rp,
                u,
                up,
                stop,
            ),
            axis=0,
        )
        # --------------
        label = np.zeros(len(testData))  # total
        pointer = 0
        value = 0
        for i in self.dataLengthList:
            label[pointer : pointer + i] = value
            pointer = pointer + i
            value = value + 1
        return testData, label

labels = [
    "B'",
    "B ",
    "D'",
    "D ",
    "F ",
    "F'",
    "L'",
    "L ",
    "R ",
    "R'",
    "U ",
    "U'",
    "Stop",
]


# evaluator = ModelEvaluator(labels)
# testdata = evaluator.testData
# testlabel = evaluator.testlabel
# model = keras.models.load_model("lstm_2hand_model.keras")
# predictions = model.predict(testdata)
# y_pred = np.argmax(predictions, axis=1)

# cm = confusion_matrix(testlabel, y_pred)

# def plot_confusion_matrix(cm, classes):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                 xticklabels=classes, yticklabels=classes)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.title('Confusion Matrix')
#     plt.show()

# plot_confusion_matrix(cm, labels)

