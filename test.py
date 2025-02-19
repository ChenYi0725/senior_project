import tools.data_organizer
import numpy as np
import keras
import time 
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt



# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.config.threading.set_inter_op_parallelism_threads(4)
resultsList = [
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

organizer = tools.data_organizer.DataOrganizer()

# model  = keras.models.load_model("lstm_index_thumb_model.keras")
# testData = organizer.preprocessForIndexaAndThumbModel(organizer.getDataFromTxt("test"))
# predictData = np.expand_dims(testData[0], axis=0)
# startTime = time.time()
# prediction = model.predict(predictData, verbose=0)
# endTime = time.time()
# print(endTime - startTime)

# model2 = keras.models.load_model("the_precious_working_model/lstm_2hand_noCTC_60Features.keras")
# testData2 = organizer.preprocessData(organizer.getDataFromTxt("test"))
# predictData2 = np.expand_dims(testData2[0], axis=0)
# startTime = time.time()
# prediction = model2.predict(predictData2, verbose=0)
# endTime = time.time()
# print(endTime - startTime)
lstmModel = keras.models.load_model(
    "LSTM_model_9moves.keras",
    # "lstm_2hand_shirnk_model.keras"
)
with open("test.txt", "r") as file:
    content = file.read()

data = eval(content)
data = np.array(data)

for i in range(len(data)):
    predictData = np.expand_dims(data[i], axis=0)
# predictData = np.expand_dims(predictData, axis=0)  # (1, timeSteps, features)
    predictData = organizer.preprocessData(predictData)
    prediction = lstmModel.predict(predictData, verbose=0)  # error
    predictedResult = np.argmax(prediction, axis=1)[0]
    probabilities = prediction[0][predictedResult]

    print(f"result:{resultsList[predictedResult]},probabilities:{probabilities}")
