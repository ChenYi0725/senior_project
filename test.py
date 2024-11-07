import tools.data_organizer
import numpy as np
import keras

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
lstmModel = keras.models.load_model(
    "lstm_2hand_shirnk_model.keras",
)
with open("result.txt", "r") as file:
    content = file.read()
predictData = [eval(content)]
predictData = np.array(predictData)
# predictData = np.expand_dims(data, axis=0)  # (1, timeSteps, features)
predictData = organizer.preprocessingForShirnkModel(predictData)
prediction = lstmModel.predict(predictData, verbose=0)  # error
predictedResult = np.argmax(prediction, axis=1)[0]
probabilities = prediction[0][predictedResult]  

print(f"result:{resultsList[predictedResult]},probabilities:{probabilities}")
