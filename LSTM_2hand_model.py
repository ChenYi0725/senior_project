import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K
import tensorflow as tf
import tools.data_oranizer as do
from keras import regularizers
import numpy as np

dataLengthList = []
organizer = do.DataOrganizer()


def initData(inputList):
    global dataLengthList
    inputList = np.array(inputList)
    dataLengthList.append(len(inputList))
    inputList = organizer.getRelativeLocation(inputList)
    return inputList


# ========================
backClockwiseData = organizer.getDataFromTxt("data_set_2hand/back_clockwise_2hands")
backCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hand/back_counter_clockwise_2hands"
)
bottomLeftData = organizer.getDataFromTxt("data_set_2hand/bottom_left_2hands")
bottomRightData = organizer.getDataFromTxt("data_set_2hand/bottom_right_2hands")
frontClockwiseData = organizer.getDataFromTxt("data_set_2hand/front_clockwise_2hands")
frontCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hand/front_counter_clockwise_2hands"
)
leftDownData = organizer.getDataFromTxt("data_set_2hand/left_down_2hands")
leftUpData = organizer.getDataFromTxt("data_set_2hand/left_up_2hands")
rightDownData = organizer.getDataFromTxt("data_set_2hand/right_down_2hands")
rightUpData = organizer.getDataFromTxt("data_set_2hand/right_up_2hands")
topLeftData = organizer.getDataFromTxt("data_set_2hand/top_left_2hands")
topRightData = organizer.getDataFromTxt("data_set_2hand/top_right_2hands")
stopData = organizer.getDataFromTxt("data_set_2hand/stop_2hands")
# ==========================
backClockwiseData = initData(backClockwiseData)
backCounterClockwiseData = initData(backCounterClockwiseData)
bottomLeftData = initData(bottomLeftData)
bottomRightData = initData(bottomRightData)
frontClockwiseData = initData(frontClockwiseData)
frontCounterClockwiseData = initData(frontCounterClockwiseData)
leftDownData = initData(leftDownData)
leftUpData = initData(leftUpData)
rightDownData = initData(rightDownData)
rightUpData = initData(rightUpData)
topLeftData = initData(topLeftData)
topRightData = initData(topRightData)
stopData = initData(stopData)
# =====================
data = np.concatenate(
    (
        backClockwiseData,
        backCounterClockwiseData,
        bottomLeftData,
        bottomRightData,
        frontClockwiseData,
        frontCounterClockwiseData,
        leftDownData,
        leftUpData,
        rightDownData,
        rightUpData,
        topLeftData,
        topRightData,
        stopData,
    ),
    axis=0,
)


print(f"whole:{len(data)}")

target = np.zeros(len(data))  # total
targetPointer = 0
targetValue = 0
for i in dataLengthList:
    target[targetPointer : targetPointer + i] = targetValue
    targetPointer = targetPointer + i
    targetValue = targetValue + 1

# print(target)
print("=====================")
# 定義模型
model = Sequential()
model.add(
    LSTM(
        243,
        activation="tanh",
        input_shape=(21, 84),  # 21,84
        kernel_regularizer=regularizers.l2(0.01),
    )
)  # LSTM層，100個神經元，每個樣本有21個時間點，42個特徵，正則化強度0.01
# ===========================================
model.add(Dense(13, activation="softmax"))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# weights = model.layers[0].get_weights()  # 改權重
# # weights[0] 為權重矩陣
# weights[0][:, 0:12] *= 0
# weights[0][:, 18:42] *= 0
# weights[0][:, 12:18] *= 2.0
# # print(f"weight{weights}")
# model.layers[0].set_weights(weights)
# ===========================================
# 訓練模型
model.fit(data, target, epochs=650, batch_size=32, verbose=2)

loss = model.evaluate(data, target)

print("loss:", loss)

model.save("lstm_2hand_model.keras")
