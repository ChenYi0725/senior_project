import tools.data_organizer
import numpy as np
import keras
import time 
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt



tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
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

organizer = tools.data_organizer.data_organizer()

model  = keras.models.load_model("lstm_index_thumb_model.keras")
testData = organizer.preprocessForIndexaAndThumbModel(organizer.get_data_from_txt("test"))
predictData = np.expand_dims(testData[0], axis=0)
startTime = time.time()
prediction = model.predict(predictData, verbose=0)
endTime = time.time()
print(endTime - startTime)

model2 = keras.models.load_model("the_precious_working_model/lstm_2hand_noCTC_60Features.keras")
testData2 = organizer.preprocess_data(organizer.get_data_from_txt("test"))
predictData2 = np.expand_dims(testData2[0], axis=0)
startTime = time.time()
prediction = model2.predict(predictData2, verbose=0)
endTime = time.time()
print(endTime - startTime)