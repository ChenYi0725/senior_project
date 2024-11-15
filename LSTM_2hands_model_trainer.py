import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import tools.model_evaluator as me

np.set_printoptions(threshold=np.inf)

timeSteps = 21
features = 60
output = 13

dataLengthList = []
organizer = do.DataOrganizer()
labelsMappingList = [
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
    "U'",
    "U ",
    "Stop",
    "wait",
]
evaluator = me.ModelEvaluator(labelsMappingList)


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


def initData(inputList):  # inputList.shape = (data numbers, time step, features)
    global dataLengthList
    inputList = np.array(inputList)
    inputList = organizer.preprocessingData(inputList)
    dataLengthList.append(len(inputList))
    return inputList


def expandTo2Hands(fingerlist):
    for i in range(len(fingerlist)):
        newFeature = fingerlist[i] + 42
        fingerlist.append(newFeature)
    return fingerlist


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
backClockwiseData = organizer.getDataFromTxt("data_set_2hands/back_clockwise_2hands")
backCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/back_counter_clockwise_2hands"
)
bottomLeftData = organizer.getDataFromTxt("data_set_2hands/bottom_left_2hands")
bottomRightData = organizer.getDataFromTxt("data_set_2hands/bottom_right_2hands")
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
print(f"label:{labels.shape}")
print("building model")
# 定義模型
inputs = layers.Input(shape=(None, features), name="input")
lstmLayer = layers.Bidirectional(
    layers.LSTM(
        256,
        activation="tanh",
        kernel_regularizer=regularizers.l2(0.01),
        return_sequences=True,
    )
)(inputs)
lstmLayer = layers.Dense(output + 1, activation="softmax")(lstmLayer)
lstmModel = keras.Model(inputs, lstmLayer)


# lstmModel = keras.models.Sequential()
# lstmModel.add(
#     layers.Bidirectional(
#         layers.LSTM(
#             units=256,
#             activation="tanh",
#             input_shape=(timeSteps, features),
#             kernel_regularizer=regularizers.l2(0.01),
#             return_sequences=True,  # for ctc
#         )
#     )
# )
# lstmModel.add(layers.Dense(output + 1, activation="softmax"))  # 13 + 1 for ctc blanky
# ========================= add ctc loss
yPred = lstmModel.output
labelsForCTC = layers.Input(name="label", shape=[None], dtype="int32")
inputLength = layers.Input(name="inputLength", shape=(1,), dtype="int32")
labelLength = layers.Input(name="labelLength", shape=(1,), dtype="int32")
ctcLoss = layers.Lambda(ctcLossFunction, output_shape=(1,), name="ctc")(
    [yPred, labelsForCTC, inputLength, labelLength]
)
ctcModel = keras.Model(
    inputs=[lstmModel.input, labelsForCTC, inputLength, labelLength], outputs=ctcLoss
)
# =========================
print("compile model")
ctcModel.compile(
    optimizer="adam",
    # loss=keras.losses.SparseCategoricalCrossentropy(),
    loss={"ctc": lambda y_true, y_pred: y_pred},
    metrics=["accuracy"],
)

# =================
# weights = model.layers[0].get_weights()  # 改權重

# thumb = [4, 5, 6, 7, 8, 9]
# indexFinger = [12, 13, 14, 15, 16, 17]
# middleFinger = [20, 21, 22, 23, 24, 25]
# ringFinger = [28, 29, 30, 31, 32, 33]
# littleFinger = [36, 37, 38, 39, 40, 41]
# palm = [0, 1, 2, 3, 10, 11, 18, 19, 26, 27, 34, 35]

# thumb = expandTo2Hands(thumb)
# indexFinger = expandTo2Hands(indexFinger)
# middleFinger = expandTo2Hands(middleFinger)
# ringFinger = expandTo2Hands(ringFinger)
# littleFinger = expandTo2Hands(littleFinger)
# palm = expandTo2Hands(palm)

# weights[0][:, thumb] *= 0.5
# weights[0][:, indexFinger] *= 1
# weights[0][:, middleFinger] *= 0.5
# weights[0][:, ringFinger] *= 1
# weights[0][:, littleFinger] *= 1
# weights[0][:, palm] *= 0
# # weights[0] 為權重矩陣
# # 左手 前42
# # 食指、無名指、小指 => 2
# # 拇指、中指 => 1
# # 手掌、手腕 => 0  0 1 5 9 13 17

# model.layers[0].set_weights(weights)
# ===========================================
# 訓練模型
print("Start Training")

# model.fit(data, labels, epochs=650, batch_size=21, verbose=1, callbacks=[evaluator])
inputLength = np.array([data.shape[1]] * len(data), dtype=np.int32)
labelLength = np.ones(len(labels), dtype=np.int32)

print("data shape:", data.shape)
print("labels shape:", labels.shape)
print("inputLength shape:", np.array(inputLength).shape)
print("labelLength shape:", np.array(labelLength).shape)
print("inputLength example:", inputLength[:10])
print("labelLength example:", labelLength[:10])
# 修改形狀====================
labels = np.expand_dims(labels, -1)
target_shape = (labels.shape[0], output + 1)
padded_labels = np.full(target_shape, labels[:, 0:1], dtype=labels.dtype)
padded_labels[:, : labels.shape[1]] = labels
labels = padded_labels


inputLength = np.expand_dims(
    np.array([data.shape[1]] * len(data), dtype=np.int32), -1
)  # (13000, 1)
labelLength = np.expand_dims(
    np.array([1] * len(data), dtype=np.int32), -1
)  # (13000, 1)

# 驗證形狀
# print("data shape:", data.shape)
# print("labels shape:", labels.shape)  # 應該要是(13000, 14)
# print("inputLength shape:", inputLength.shape)
# print("labelLength shape:", labelLength.shape)


# =================
ctcModel.fit(  
    [data, labels, inputLength, labelLength],
    labels,
    epochs=350,
    batch_size=21,
    verbose=1,
    callbacks=[evaluator],
)

# evaluateModel(ctcModel, data, labels, inputLength, labelLength)
print("save model")
# 輸出模型
# exportSavedModelAndTflite(model)
ctcModel.save("lstm_2hand_model.keras")
ctcModel.save("lstm_2hand_model.h5")
print("finish")
