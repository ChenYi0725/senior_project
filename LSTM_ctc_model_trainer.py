import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

time_steps = 21
features = 60
output = 13

data_length_list = []
organizer = do.data_organizer()
labels_mapping_list = [
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
evaluator = me.Model_Evaluator(labels_mapping_list)

def ctc_loss_function(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def init_data(inputList):  # inputList.shape = (data numbers, time step, features)
    global data_length_list
    inputList = np.array(inputList)
    inputList = organizer.preprocess_data(inputList)
    data_length_list.append(len(inputList))
    return inputList


def expand_to_2hands(finger_list):
    for i in range(len(finger_list)):
        new_feature = finger_list[i] + 42
        finger_list.append(new_feature)
    return finger_list


def export_saved_model_and_tflite(model):
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


def evaluate_model(model, data, labels, input_length, label_length):
    loss = model.evaluate([data, labels, input_length, label_length], verbose=1)
    print("loss:", loss)


# ========================
print("loading data")
back_clockwise_data = organizer.get_data_from_txt("data_set_2hands/back_clockwise_2hands")
back_counter_clockwise_data = organizer.get_data_from_txt(
    "data_set_2hands/back_counter_clockwise_2hands"
)
bottom_left_data = organizer.get_data_from_txt("data_set_2hands/bottom_left_2hands")
bottom_right_data = organizer.get_data_from_txt("data_set_2hands/bottom_right_2hands")
front_clockwise_data = organizer.get_data_from_txt("data_set_2hands/front_clockwise_2hands")
front_counter_clockwise_data = organizer.get_data_from_txt(
    "data_set_2hands/front_counter_clockwise_2hands"
)
left_down_data = organizer.get_data_from_txt("data_set_2hands/left_down_2hands")
left_up_data = organizer.get_data_from_txt("data_set_2hands/left_up_2hands")
right_down_data = organizer.get_data_from_txt("data_set_2hands/right_down_2hands")
right_up_data = organizer.get_data_from_txt("data_set_2hands/right_up_2hands")
top_left_data = organizer.get_data_from_txt("data_set_2hands/top_left_2hands")
top_right_data = organizer.get_data_from_txt("data_set_2hands/top_right_2hands")
stop_data = organizer.get_data_from_txt("data_set_2hands/stop_2hands")

# ==========================
print("init Data")
back_clockwise_data = init_data(back_clockwise_data)
back_counter_clockwise_data = init_data(back_counter_clockwise_data)
bottom_left_data = init_data(bottom_left_data)
bottom_right_data = init_data(bottom_right_data)
front_clockwise_data = init_data(front_clockwise_data)
front_counter_clockwise_data = init_data(front_counter_clockwise_data)
left_down_data = init_data(left_down_data)
left_up_data = init_data(left_up_data)
right_down_data = init_data(right_down_data)
right_up_data = init_data(right_up_data)
top_left_data = init_data(top_left_data)
top_right_data = init_data(top_right_data)
stop_data = init_data(stop_data)
# =====================


data = np.concatenate(
    (
        back_clockwise_data,
        back_counter_clockwise_data,
        bottom_left_data,
        bottom_right_data,
        front_clockwise_data,
        front_counter_clockwise_data,
        left_down_data,
        left_up_data,
        right_down_data,
        right_up_data,
        top_left_data,
        top_right_data,
        stop_data,
    ),
    axis=0,
)

print(f"training data shape:{data.shape}")

print(f"data len:{len(data)}")
# =========================訓練集label 0 開始
labels = np.zeros(len(data), dtype=np.int32)  # total
labels_pointer = 0
labels_value = 0
for i in data_length_list:
    labels[labels_pointer : labels_pointer + i] = labels_value
    labels_pointer = labels_pointer + i
    labels_value = labels_value + 1
# =========================
print(f"label:{labels.shape}")
print("building model")
# 定義模型
inputs = layers.Input(shape=(None, features), name="input")
lstm_layer = layers.Bidirectional(
    layers.LSTM(
        256,
        activation="tanh",
        kernel_regularizer=regularizers.l2(0.01),
        return_sequences=True,
    )
)(inputs)
lstm_layer = layers.Dense(output + 1, activation="softmax")(lstm_layer)
lstm_model = keras.Model(inputs, lstm_layer)


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
y_pred = lstm_model.output
labels_for_ctc = layers.Input(name="label", shape=[None], dtype="int32")
input_length = layers.Input(name="inputLength", shape=(1,), dtype="int32")
label_length = layers.Input(name="labelLength", shape=(1,), dtype="int32")
ctc_loss = layers.Lambda(ctc_loss_function, output_shape=(1,), name="ctc")(
    [y_pred, labels_for_ctc, input_length, label_length]
)
ctc_model = keras.Model(
    inputs=[lstm_model.input, labels_for_ctc, input_length, label_length], outputs=ctc_loss
)
# =========================
print("compile model")
ctc_model.compile(
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
input_length = np.array([data.shape[1]] * len(data), dtype=np.int32)
label_length = np.ones(len(labels), dtype=np.int32)

print("data shape:", data.shape)
print("labels shape:", labels.shape)
print("inputLength shape:", np.array(input_length).shape)
print("labelLength shape:", np.array(label_length).shape)
print("inputLength example:", input_length[:10])
print("labelLength example:", label_length[:10])
# 修改形狀====================
labels = np.expand_dims(labels, -1)
target_shape = (labels.shape[0], output + 1)
padded_labels = np.full(target_shape, labels[:, 0:1], dtype=labels.dtype)
padded_labels[:, : labels.shape[1]] = labels
labels = padded_labels


input_length = np.expand_dims(
    np.array([data.shape[1]] * len(data), dtype=np.int32), -1
)  # (13000, 1)
label_length = np.expand_dims(
    np.array([1] * len(data), dtype=np.int32), -1
)  # (13000, 1)

# 驗證形狀
# print("data shape:", data.shape)
# print("labels shape:", labels.shape)  # 應該要是(13000, 14)
# print("inputLength shape:", inputLength.shape)
# print("labelLength shape:", labelLength.shape)


# =================
ctc_model.fit(  # 收到none 值，尚未找出原因->遞迴測labels, inputLength, labelLength
    [data, labels, input_length, label_length],
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
ctc_model.save("lstm_2hand_model.keras")
ctc_model.save("lstm_2hand_model.h5")
print("finish")
