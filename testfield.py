import data_oranizer as do
import numpy as np
import random

# organizer = do.DataOrganizer()
# downTestData = organizer.getDataFromTxt("test_down")
# downTestData = np.array(downTestData)
# downTestData = organizer.getRelativeWithFirstTimeStep(downTestData)


# inputData = []
# sample = []
# feature1 = 1045
# feature2 = 1048
# feature3 = 45  # 1~10
# feature4 = 8  # 11~20
# for j in range(10):
#     amplitude = random.randint(1, 2)
#     features = [
#         feature1 + (j * amplitude),
#         feature2 + (j * amplitude),
#         feature3,
#         feature4,
#     ]
#     sample.append(features)
# inputData.append(sample)

organizer = do.DataOrganizer()
downTestData = organizer.getDataFromTxt("test_down")
for i in range(len(downTestData)):
    predictData = downTestData[i]
    predictData = np.array(predictData)
    print(predictData.shape)
downTestData = organizer.getRelativeWithFirstTimeStep(downTestData)
# al = [inputData[0]]

# print(len(inputData))
# print(len(inputData[0]))
# print(len(inputData[0][0]))
# prediction = model.predict(inputData)
# predictedResult = np.argmax(prediction)
# model.predict(inputData)
# print(predictedResult)
