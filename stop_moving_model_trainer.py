import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me
import random
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)

timeSteps = 21
features = 36
output = 2

dataLengthList = []
organizer = do.DataOrganizer()
labelsMappingList = [
    "move",
    "Stop",
]


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


def initData(inputList):  # inputList.shape = (data numbers, time step, features)
    global dataLengthList
    inputList = np.array(inputList)
    inputList = organizer.preprocessForShirnkModel(inputList)
    dataLengthList.append(len(inputList))
    return inputList


def exportSavedModelAndTflite(model):
    model.export(filepath="lstm_2hand_saved_model", format="tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("lstm_2hand_saved_model")
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open("lstm_2hand.tflite", "wb") as f:
        f.write(tflite_model)


def evaluateModel(model, data, labels, inputLength, labelLength):
    loss = model.evaluate([data, labels, inputLength, labelLength], verbose=1)
    print("loss:", loss)


# ========================
print("loading data")
data_paths = [
    ("frontClockwiseData", "data_set_2hands/front_clockwise_2hands"),
    ("frontCounterClockwiseData", "data_set_2hands/front_counter_clockwise_2hands"),
    ("leftDownData", "data_set_2hands/left_down_2hands"),
    ("leftUpData", "data_set_2hands/left_up_2hands"),
    ("rightDownData", "data_set_2hands/right_down_2hands"),
    ("rightUpData", "data_set_2hands/right_up_2hands"),
    ("topLeftData", "data_set_2hands/top_left_2hands"),
    ("topRightData", "data_set_2hands/top_right_2hands"),
    ("stopData", "data_set_2hands/stop_2hands"),
]
data_dict = {}
for name, path in tqdm(data_paths, desc="Loading Data"):
    data_dict[name] = organizer.getDataFromTxt(path)

frontClockwiseData = data_dict["frontClockwiseData"]
frontCounterClockwiseData = data_dict["frontCounterClockwiseData"]
leftDownData = data_dict["leftDownData"]
leftUpData = data_dict["leftUpData"]
rightDownData = data_dict["rightDownData"]
rightUpData = data_dict["rightUpData"]
topLeftData = data_dict["topLeftData"]
topRightData = data_dict["topRightData"]
stopData = data_dict["stopData"]
# ===============
moveData = np.concatenate(
    (
        np.array(random.sample(frontClockwiseData, 125)),
        np.array(random.sample(frontCounterClockwiseData, 125)),
        np.array(random.sample(leftDownData, 125)),
        np.array(random.sample(leftUpData, 125)),
        np.array(random.sample(rightDownData, 125)),
        np.array(random.sample(rightUpData, 125)),
        np.array(random.sample(topLeftData, 125)),
        np.array(random.sample(topRightData, 125)),
    ),
    axis=0,
)

# ==========================
print("init Data")
moveData = initData(moveData)
stopData = initData(stopData)
# =====================


data = np.concatenate(
    (
        moveData,
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
    epochs=350,
    batch_size=21,
    verbose=1,
)

print("save model")
# 輸出模型

model.save("lstm_moving_dectect_model.keras")
print("finish")
