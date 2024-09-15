import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tools import data_organizer as do


class ModelEvaluator(Callback):
    def __init__(self, label):
        super(ModelEvaluator, self).__init__()
        self.organizer = do.DataOrganizer()
        self.losses = []
        self.label = label
        self.dataLengthList = []
        self.xTest, self.yTest = self.createTestData()
        print(self.xTest.shape)
        print(self.yTest.shape)

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
        yPred = self.model.predict(self.xTest)
        yPredClasses = np.argmax(yPred, axis=1)
        yTrue = np.argmax(self.yTest, axis=1)

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


# # 生成三個類別的數據
# num_samples_per_class = 300
# num_features = 2  # 使用二維特徵以便可視化
# num_classes = 3

# # 隨機生成數據
# np.random.seed(0)
# X, y = make_classification(
#     n_samples=num_samples_per_class * num_classes,
#     n_features=num_features,
#     n_informative=num_features,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     n_classes=num_classes,
#     class_sep=2,  # 增加類別間距
#     random_state=0,
# )

# # 將標籤轉換為 one-hot 編碼
# y_one_hot = to_categorical(y, num_classes)

# # 分割數據集
# num_train_samples = num_samples_per_class * (num_classes - 1)  # 用於訓練的樣本數量
# x_train = X[:num_train_samples]
# y_train = y_one_hot[:num_train_samples]

# x_test = X[num_train_samples:]
# y_test = y_one_hot[num_train_samples:]

# # 定義模型
# model = keras.models.Sequential(
#     [
#         keras.layers.Dense(64, activation="relu", input_shape=(num_features,)),
#         keras.layers.Dense(num_classes, activation="softmax"),
#     ]
# )
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # 顯示的標籤
# labels = [f"Class {i}" for i in range(num_classes)]
# # 創建評估器實例
# evaluator = ModelEvaluator(x_test, y_test, labels)
# # 訓練模型
# model.fit(x_train, y_train, epochs=10, callbacks=[evaluator])
