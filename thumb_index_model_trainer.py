import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

time_steps = 21
features = 20
output = 9

data_length_list = []
organizer = do.data_organizer()
labels_mapping_list = [
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




def init_data(input_list):  # inputList.shape = (data numbers, time step, features)
    global data_length_list
    input_list = np.array(input_list)
    input_list = organizer.preprocessForIndexaAndThumbModel(input_list)
    data_length_list.append(len(input_list))
    return input_list



def evaluate_model(model, data, labels, input_length, label_length):
    loss = model.evaluate([data, labels, input_length, label_length], verbose=1)
    print("loss:", loss)


# ========================
print("loading data")
# backClockwiseData = organizer.getDataFromTxt("data_set_2hands/back_clockwise_2hands")
# backCounterClockwiseData = organizer.getDataFromTxt(
#     "data_set_2hands/back_counter_clockwise_2hands"
# )
# bottomLeftData = organizer.getDataFromTxt("data_set_2hands/bottom_left_2hands")
# bottomRightData = organizer.getDataFromTxt("data_set_2hands/bottom_right_2hands")
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
# backClockwiseData = initData(backClockwiseData)
# backCounterClockwiseData = initData(backCounterClockwiseData)
# bottomLeftData = initData(bottomLeftData)
# bottomRightData = initData(bottomRightData)
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
        # backClockwiseData,
        # backCounterClockwiseData,
        # bottomLeftData,
        # bottomRightData,
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
        units=64,
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
