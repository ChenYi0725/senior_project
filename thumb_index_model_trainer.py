import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

timeSteps = 21
features = 20
output = 9

dataLengthList = []
organizer = do.DataOrganizer()
labelsMappingList = [
    # "B'",
    # "B ",
    # "D'",
    # "D ",
    "F ",
    "F'",
    "L'",
    "L ",
    "R ",
    "R'",
    "U'",
    "U ",
    "Stop",
    "wait",
]
# evaluator = me.ModelEvaluator(labelsMappingList)


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


def initData(inputList):  # inputList.shape = (data numbers, time step, features)
    global dataLengthList
    inputList = np.array(inputList)
    inputList = organizer.preprocessForIndexaAndThumbModel(inputList)
    dataLengthList.append(len(inputList))
    return inputList



def evaluateModel(model, data, labels, inputLength, labelLength):
    loss = model.evaluate([data, labels, inputLength, labelLength], verbose=1)
    print("loss:", loss)


# ========================
print("loading data")
# backClockwiseData = organizer.getDataFromTxt("data_set_2hands/back_clockwise_2hands")
# backCounterClockwiseData = organizer.getDataFromTxt(
#     "data_set_2hands/back_counter_clockwise_2hands"
# )
# bottomLeftData = organizer.getDataFromTxt("data_set_2hands/bottom_left_2hands")
# bottomRightData = organizer.getDataFromTxt("data_set_2hands/bottom_right_2hands")
frontClockwiseData = organizer.getDataFromTxt("data_set_2hands/front_clockwise_2hands")
frontCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/front_counter_clockwise_2hands"
)
leftDownData = organizer.getDataFromTxt("data_set_2hands/left_down_2hands")
leftUpData = organizer.getDataFromTxt("data_set_2hands/left_up_2hands")
rightDownData = organizer.getDataFromTxt("data_set_2hands/right_down_2hands")
rightUpData = organizer.getDataFromTxt("data_set_2hands/right_up_2hands")
topLeftData = organizer.getDataFromTxt("data_set_2hands/top_left_2hands")
topRightData = organizer.getDataFromTxt("data_set_2hands/top_right_2hands")
stopData = organizer.getDataFromTxt("data_set_2hands/stop_2hands")

# ==========================
print("init Data")
# backClockwiseData = initData(backClockwiseData)
# backCounterClockwiseData = initData(backCounterClockwiseData)
# bottomLeftData = initData(bottomLeftData)
# bottomRightData = initData(bottomRightData)
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
        # backClockwiseData,
        # backCounterClockwiseData,
        # bottomLeftData,
        # bottomRightData,
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

print(f"training data shape:{data.shape}")

print(f"data len:{len(data)}")
# =========================訓練集label 0 開始
labels = np.zeros(len(data), dtype=np.int32)  # total
labelsPointer = 0
labelsValue = 0
for i in dataLengthList:
    labels[labelsPointer : labelsPointer + i] = labelsValue
    labelsPointer = labelsPointer + i
    labelsValue = labelsValue + 1
# =========================
print("building model")
model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(
        units=64,
        activation="tanh",
        input_shape=(timeSteps, features),
        kernel_regularizer=regularizers.l2(0.01),
    )
)
model.add(keras.layers.Dense(output, activation="softmax"))


# =========================
print("compile model")
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


# 訓練模型
print("Start Training")

# =================
model.fit(
    data,
    labels,
    epochs=650,
    batch_size=21,
    verbose=1,
    # callbacks=[evaluator],
)

# evaluateModel(ctcModel, data, labels, inputLength, labelLength)
print("save model")
# 輸出模型
# exportSavedModelAndTflite(model)
model.save("lstm_index_thumb_model.keras")
model.save("lstm_2hand_shirnk_model.h5")
print("finish")