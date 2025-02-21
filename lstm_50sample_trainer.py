import keras
import tensorflow as tf
import tools.data_organizer as do
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

time_steps = 21
features = 60
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
# evaluator = me.ModelEvaluator(labelsMappingList)


def ctc_loss_function(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def init_data(input_list):  # inputList.shape = (data numbers, time step, features)
    global data_length_list
    input_list = np.array(input_list[0:100])
    input_list = organizer.preprocess_data(input_list)
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
print("building model")
model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(
        units=8,
        activation="tanh",
        input_shape=(time_steps, features),
        # kernel_regularizer=regularizers.l2(0.01),
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
    epochs=50,
    batch_size=21,
    verbose=1,
    # callbacks=[evaluator],
)

# evaluateModel(ctcModel, data, labels, inputLength, labelLength)
print("save model")
# 輸出模型
# exportSavedModelAndTflite(model)
model.save("LSTM_model_9moves_50sample.keras")
# model.save("lstm_2hand_shirnk_model.h5")
print("finish")
