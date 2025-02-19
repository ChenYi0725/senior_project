import tensorflow as tf

import keras

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.fashion_mnist.load_data()

test_model = keras.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

test_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

test_model.fit(train_images, train_labels, epochs=1)

test_model = keras.models.load_model("lstm_2hand_save_model")

test_model.export(
    "test", "tf_saved_model"
)  # replace tf.saved_model.save with this line

converter = tf.lite.TFLiteConverter.from_saved_model("test")
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
