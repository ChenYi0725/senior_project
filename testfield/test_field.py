import numpy as np
import tools.data_oranizer as do

inputList = [[[1, 2, 4, 6, 8.0], [3, 4, 5, 6, 7]], [[5, 6, 7, 8, 9], [7, 8, 9, 10, 11]]]

o = do.DataOrganizer()

inputList = o.preprocessingData(inputList)
inputList = o.normalizedWithEachTimeSteps(inputList)
# inputList = np.array(inputList)

# for i in range(len(inputList)):
#     for j in range(inputList.shape[i]):
#         inputList[i, j] = (inputList[i, j] - inputList[i, j].min()) / (
#             inputList[i, j].max() - inputList[i, j].min()
#         )

print(inputList)
