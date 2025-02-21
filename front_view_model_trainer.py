import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

time_steps = 21
features = 36
output = 9

data_length_list = []
organizer = do.data_organizer()
labels_mapping_list = [
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



def ctc_loss_function(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def init_data(input_list):  # inputList.shape = (data numbers, time step, features)
    global data_length_list
    input_list = np.array(input_list)
    input_list = organizer.preprocess_for_shirnk_model(input_list)
    data_length_list.append(len(input_list))
    return input_list


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
f_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/F")
fp_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/F'")
lp_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/L'")
l_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/L")
r_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/R")
rp_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/R'")
up_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/U'")
u_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/U")
stop_data = organizer.get_data_from_txt("front_view_dataset/bright_dataset/stop")

# ==========================
print("init Data")

f_data = init_data(f_data)
fp_data = init_data(fp_data)
lp_data = init_data(lp_data)
l_data = init_data(l_data)
r_data = init_data(r_data)
rp_data = init_data(rp_data)
up_data = init_data(up_data)
u_data = init_data(u_data)
stop_data = init_data(stop_data)
# =====================


data = np.concatenate(
    (f_data, fp_data, lp_data, l_data, r_data, rp_data, up_data, u_data,stop_data),
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
print("building model")
model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(
        units=16,
        activation="tanh",
        input_shape=(time_steps, features),
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
