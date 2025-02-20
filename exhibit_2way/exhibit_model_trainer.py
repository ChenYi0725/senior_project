import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me

np.set_printoptions(threshold=np.inf)

time_steps = 21
features = 42
output = 3

data_length_list = []
organizer = do.data_organizer()
labels_mapping_list = [
"horizontal",
"vertical",
"stop"
]


def init_data(input_list):  # inputList.shape = (data numbers, time step, features)
    global data_length_list
    input_list = np.array(input_list)
    input_list = organizer.preprocess_exhibit_data(input_list)
    data_length_list.append(len(input_list))
    return input_list



# ========================
print("loading data")
horizontal_data = organizer.get_data_from_txt("exhibit_data_set/horizontal")
vertical_data = organizer.get_data_from_txt("exhibit_data_set\\vertical")
stop_data = organizer.get_data_from_txt("exhibit_data_set/stop")




# ==========================
print("init Data")
# backClockwiseData = initData(backClockwiseData)
# backCounterClockwiseData = initData(backCounterClockwiseData)
# bottomLeftData = initData(bottomLeftData)
# bottomRightData = initData(bottomRightData)
horizontal_data = init_data(horizontal_data)
vertical_data = init_data(vertical_data)
stop_data = init_data(stop_data)
# =====================


data = np.concatenate(
    (
        horizontal_data,
        vertical_data,
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
    batch_size=64,
    verbose=1,
    # callbacks=[evaluator],
)

# evaluateModel(ctcModel, data, labels, inputLength, labelLength)
print("save model")

model.save("exhibit_model.keras")
print("finish")
