import tools.data_organizer
import numpy as np
import keras
import time 
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt



tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
results_list = [
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
test_data = organizer.preprocessForIndexaAndThumbModel(organizer.get_data_from_txt("test"))
predict_data = np.expand_dims(test_data[0], axis=0)
start_time = time.time()
prediction = model.predict(predict_data, verbose=0)
end_time = time.time()
print(end_time - start_time)

model_2 = keras.models.load_model("the_precious_working_model/lstm_2hand_noCTC_60Features.keras")
test_data_2 = organizer.preprocess_data(organizer.get_data_from_txt("test"))
predict_data_2 = np.expand_dims(test_data_2[0], axis=0)
start_time = time.time()
prediction = model_2.predict(predict_data_2, verbose=0)
end_time = time.time()
print(end_time - start_time)