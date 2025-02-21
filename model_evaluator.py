import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import data_organizer as do
from tools import data_organizer as do


class Model_Evaluator(Callback):
    def __init__(self, label):
        print("loading test data for evaluator")
        super(Model_Evaluator, self).__init__()
        self.organizer = do.data_organizer()
        self.losses = []
        self.accuracies = []
        self.label = label
        self.data_length_list = []
        self.x_test, self.y_test = self.createTestData()

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accuracies.append(logs.get("accuracy"))


    def on_train_end(self, logs=None):
        self.drawLossFunction()
        self.draw_confusion_matrix()
        self.draw_accuracy()

    def drawLossFunction(self, logs={}):
        # 繪製損失值圖表
        plt.plot(self.losses)
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def draw_confusion_matrix(self):
        y_pred = self.model.predict(self.testData)
        y_pred_classes = np.argmax(y_pred, axis=1)

        y_true = np.argmax(self.testlabel, axis=1)

        conf_matrix = confusion_matrix(self.yTest, y_pred_classes)

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

    def draw_accuracy(self,logs={}):
        plt.plot(self.accuracies)
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()

    # ----------------------------------------------------
    def init_data(
        self, input_list
    ):  # inputList.shape = (data numbers, time step, features)
        input_list = np.array(input_list)
        self.data_length_list.append(len(input_list))
        input_list = self.organizer.preprocess_for_shirnk_model(input_list)
        return input_list

    def createTestData(self):
        # bp = self.organizer.getDataFromTxt("test_data_set/b'_test")
        # b = self.organizer.getDataFromTxt("test_data_set/b_test")
        # dp = self.organizer.getDataFromTxt("test_data_set/d'_test")
        # d = self.organizer.getDataFromTxt("test_data_set/d_test")
        f = self.organizer.get_data_from_txt("test_data_set/f_test")
        fp = self.organizer.get_data_from_txt("test_data_set/f'_test")
        lp = self.organizer.get_data_from_txt("test_data_set/l'_test")
        l = self.organizer.get_data_from_txt("test_data_set/l_test")
        r = self.organizer.get_data_from_txt("test_data_set/r_test")
        rp = self.organizer.get_data_from_txt("test_data_set/r'_test")
        u = self.organizer.get_data_from_txt("test_data_set/u_test")
        up = self.organizer.get_data_from_txt("test_data_set/u'_test")
        stop = self.organizer.get_data_from_txt("test_data_set/stop_test")
        # ---
        # bp = self.initData(bp)
        # b = self.initData(b)
        # dp = self.initData(dp)
        # d = self.initData(d)
        f = self.init_data(f)
        fp = self.init_data(fp)
        lp = self.init_data(lp)
        l = self.init_data(l)
        r = self.init_data(r)
        rp = self.init_data(rp)
        u = self.init_data(u)
        up = self.init_data(up)
        stop = self.init_data(stop)
        # ------
        testData = np.concatenate(
            (
                # bp,
                # b,
                # dp,
                # d,
                f,
                fp,
                lp,
                l,
                r,
                rp,
                up,
                u,
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

