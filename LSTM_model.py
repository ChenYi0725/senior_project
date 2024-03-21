import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K
import tensorflow as tf
import data_oranizer as do
from keras import regularizers
import numpy as np

# predict結果出問題，輸入資料交換後仍只輸出0，先檢查predict result那塊


def myLossFunction(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    feature_weights = tf.constant([1.0] * 42)  # 初始化所有權重為1
    # 對特徵進行加權
    feature_weights = tf.tensor_scatter_nd_update(
        feature_weights,
        [[12], [13], [14], [15], [16], [17]],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    )

    loss_weighted = loss * feature_weights

    # 計算加權後的損失
    loss_mean = K.mean(loss_weighted, axis=-1)
    return loss_mean


organizer = do.DataOrganizer()
downData = organizer.getDataFromTxt("down_data")
stopData = organizer.getDataFromTxt("stop_data")
leftData = organizer.getDataFromTxt("left_data")


print(f"down:{len(downData)}")
print(f"stop:{len(stopData)}")
print(f"left:{len(leftData)}")

# print(downData)
downData = np.array(downData)
stopData = np.array(stopData)
leftData = np.array(leftData)

downData = organizer.getRelativeWithFirstTimeStep(downData)
stopData = organizer.getRelativeWithFirstTimeStep(stopData)
leftData = organizer.getRelativeWithFirstTimeStep(leftData)
# data格式 eg.[3][2][1]三個樣本 兩個時間步長 一個特徵點
data = np.concatenate((stopData, downData, leftData), axis=0)

print(f"whole:{len(data)}")
target = np.zeros(900)  # total
target[:300] = 0  # down
target[300:600] = 1  # stop
target[600:] = 2  # left

# print(target)
print("=====================")
# 定義模型
model = Sequential()
model.add(
    LSTM(
        150,
        activation="relu",
        input_shape=(21, 42),  # 21,42
        kernel_regularizer=regularizers.l2(0.01),
    )
)  # LSTM層，100個神經元，每個樣本有21個時間點，42個特徵，正則化強度0.01
# ===========================================
model.add(Dense(3, activation="softmax"))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

weights = model.layers[0].get_weights()  # 改食指權重
weights[0][:, 12:18] *= 2.0
model.layers[0].set_weights(weights)
# model.compile(optimizer="adam", loss=myLossFunction, metrics=["accuracy"])
# model.add(Dense(1))  # 全連接層，輸出1個值
# 編譯模型
# model.compile(optimizer="adam", loss=myLossFunction)
# ===========================================
# 訓練模型
model.fit(data, target, epochs=500, batch_size=32, verbose=2)

loss = model.evaluate(data, target)

print("loss:", loss)


# input("continue to test down data")
# downTestData = organizer.getDataFromTxt("test_down")
# downTestData = organizer.getRelativeWithFirstTimeStep(downTestData)
# for i in range(len(downTestData)):
#     predictData = [downTestData[i]]
#     predictData = np.array(predictData)
#     prediction = model.predict(predictData)
#     predictedResult = np.argmax(prediction, axis=1)
#     print(f"prediction:{prediction}")
#     print(f"result:{predictedResult},answer:0")

# input("continue to test stop data")
# stopTestData = organizer.getDataFromTxt("test_stop")
# stopTestData = organizer.getRelativeWithFirstTimeStep(stopTestData)
# for i in range(len(stopTestData)):
#     predictData = [stopTestData[i]]
#     predictData = np.array(predictData)
#     prediction = model.predict(predictData)
#     predictedResult = np.argmax(prediction, axis=1)
#     print(f"prediction:{prediction}")
#     print(f"result:{predictedResult},answer:1")


# input("continue to test left data")
# leftTestData = organizer.getDataFromTxt("test_left")
# leftTestData = organizer.getRelativeWithFirstTimeStep(leftTestData)
# for i in range(len(leftTestData)):
#     predictData = [leftTestData[i]]
#     predictData = np.array(predictData)
#     prediction = model.predict(predictData)
#     predictedResult = np.argmax(prediction)
#     print(f"prediction:{prediction}")
#     print(f"result:{predictedResult},answer:2")
# print("done")

model.save("lstm_hand_model.h5")
