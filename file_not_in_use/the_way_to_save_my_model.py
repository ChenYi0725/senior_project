import keras
import tensorflow as tf
import tools.data_organizer as do
from keras import regularizers
from keras import layers
import numpy as np
import model_evaluator as me
import tools.data_organizer as do

organizer = do.DataOrganizer()

features = 60
output = 13


def ctcLossFunction(args):
    yPred, labels, inputLength, labelLength = args
    return tf.keras.backend.ctc_batch_cost(labels, yPred, inputLength, labelLength)


lstmModel = keras.models.load_model(
    "lstm_2hand_model.h5",
    custom_objects={"ctcLossFunction": ctcLossFunction},
    compile=False,
)


print("node 1")
print(lstmModel.summary())

inputs = layers.Input(shape=(21, features), name="input")
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


lstmModel.load_weights("lstm_2hand_model.keras")
print("load weight fin")
test_data = organizer.getDataFromTxt("test_data_set/b_test")
test_data = test_data[0]

test_data = np.expand_dims(test_data, axis=0)
input_test_data = organizer.preprocessData(test_data)
prediction = lstmModel.predict(input_test_data, verbose=0)

predictedResult = np.argmax(prediction[0], axis=1)[0]
probabilities = prediction[0][predictedResult]
print(prediction)
print(predictedResult)
# print(probabilities)
