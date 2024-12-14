import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

timeSteps = 21
features = 36
output = 9

dataLengthList = []
organizer = do.DataOrganizer()
labelsMappingList = [
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
FData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/F")
FPData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/F'")
LPData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/L'")
LData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/L")
RData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/R")
RPData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/R'")
UPData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/U'")
UData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/U")
stopData = organizer.getDataFromTxt("front_view_dataset/bright_dataset/stop")

# ==========================
print("init Data")

FData = initData(FData)
FPData = initData(FPData)
LPData = initData(LPData)
LData = initData(LData)
RData = initData(RData)
RPData = initData(RPData)
UPData = initData(UPData)
UData = initData(UData)
stopData = initData(stopData)
# =====================


data = np.concatenate(
    (FData, FPData, LPData, LData, RData, RPData, UPData, UData,stopData),
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
    batch_size=21,
    verbose=1,
)


print("save model")
# 輸出模型
# exportSavedModelAndTflite(model)
model.save("front_view_9moves_model.keras")
# model.save("lstm_2hand_shirnk_model.h5")
print("finish")
