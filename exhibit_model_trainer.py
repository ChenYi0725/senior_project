import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

timeSteps = 21
features = 42
output = 3

dataLengthList = []
organizer = do.DataOrganizer()
labelsMappingList = [
"horizontal",
"vertical",
"stop"
]


def initData(inputList):  # inputList.shape = (data numbers, time step, features)
    global dataLengthList
    inputList = np.array(inputList)
    inputList = organizer.preprocessExhibitData(inputList)
    dataLengthList.append(len(inputList))
    return inputList



# ========================
print("loading data")
horizontalData = organizer.getDataFromTxt("exhibit_data_set/horizontal")
verticalData = organizer.getDataFromTxt("exhibit_data_set\\vertical")
stopData = organizer.getDataFromTxt("exhibit_data_set/stop")




# ==========================
print("init Data")
# backClockwiseData = initData(backClockwiseData)
# backCounterClockwiseData = initData(backCounterClockwiseData)
# bottomLeftData = initData(bottomLeftData)
# bottomRightData = initData(bottomRightData)
horizontalData = initData(horizontalData)
verticalData = initData(verticalData)
stopData = initData(stopData)
# =====================


data = np.concatenate(
    (
        horizontalData,
        verticalData,
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
        units=16,
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
    epochs=300,
    batch_size=64,
    verbose=1,
    # callbacks=[evaluator],
)

# evaluateModel(ctcModel, data, labels, inputLength, labelLength)
print("save model")

model.save("exhibit_model.keras")
print("finish")
