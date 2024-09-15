import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
import numpy as np
import tools.ctc_loss_function as CTC
import tools.model_evaluator as me

timeSteps = 21
features = 60
dataLengthList = []
organizer = do.DataOrganizer()
CTCLoss = CTC.CTCLoss()
labels = [
    "B'(Back Clockwise)",
    "B (Back Counter Clockwise)",
    "D'(Bottom Left)",
    "D (Bottom Right)",
    "F (Front Clockwise)",
    "F' (Front Counter Clockwise)",
    "L'(Left Down)",
    "L (Left Up)",
    "R (Right Down)",
    "R'(Right Up)",
    "U (Top Left)",
    "U'(Top Right)",
    "Stop",
]
evaluator = me.ModelEvaluator(labels)


def initData(inputList):  # inputList.shape = (data numbers, time step, features)
    global dataLengthList
    inputList = np.array(inputList)
    inputList = organizer.preprocessingData(inputList)
    print(f"init data len{inputList.shape}")
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


def evaluateModel(model):
    loss = model.evaluate(data, target)
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

# ====================
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

print(data.shape)

print(f"data len:{len(data)}")
# ----------------------訓練集label 0 開始
target = np.zeros(len(data))  # total
targetPointer = 0
targetValue = 0
for i in dataLengthList:
    target[targetPointer : targetPointer + i] = targetValue
    targetPointer = targetPointer + i
    targetValue = targetValue + 1

print("=====================")
# ========================


evaluator = me.ModelEvaluator(labels)


# =========================
# 定義模型
model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(
        units=256,
        activation="tanh",
        input_shape=(timeSteps, features),
        kernel_regularizer=regularizers.l2(0.01),
    )
)
model.add(keras.layers.Dense(13, activation="softmax"))
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss= keras.losses.CategoricalCrossentropy(),
    # loss=CTCLoss,
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

model.fit(data, target, epochs=650, batch_size=21, verbose=1, callbacks=[evaluator])

evaluateModel(model)
print("save model")
# 輸出模型
# exportSavedModelAndTflite(model)
model.save("lstm_2hand_model.keras")
model.save("lstm_2hand_model.h5")
print("finish")
