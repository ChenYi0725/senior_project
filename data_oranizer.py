import numpy as np

test = np.array(
    [
        [[1, 2, 4, 97], [10, 28, 38, 35], [49, -84, -23, 66]],
        [[82, 55, 88, 72], [46, 42, 99, -98], [87, 23, 78, 28]],
    ]
)


class DataOrganizer:
    def cutFirstTimeStep(self, npArray):
        npArray = npArray[:, 1:, :]
        return npArray

    def getRelativeLocation(self, npArray):
        for i in range(len(npArray)):
            for j in range(len(npArray[i])):
                originX = npArray[i][j][0]
                originY = npArray[i][j][1]
                for k in range(len(npArray[i][j])):
                    if k % 2 == 0:
                        npArray[i][j][k] = npArray[i][j][k] - originX
                    else:
                        npArray[i][j][k] = npArray[i][j][k] - originY
        return npArray

    def getRelativeWithFirstTimeStep(self, npArray):
        for i in range(len(npArray)):
            originX = npArray[i][0][0]
            originY = npArray[i][0][1]
            for j in range(len(npArray[i])):
                for k in range(len(npArray[i][j])):
                    if k % 2 == 0:
                        npArray[i][j][k] = npArray[i][j][k] - originX
                    else:
                        npArray[i][j][k] = npArray[i][j][k] - originY
        return npArray

    def getDataFromTxt(self, fileName):
        with open(f"{fileName}.txt", "r") as file:
            content = file.read()
        result = eval(content)
        return result

    # def cutInhomogeneousData(self,inputList, size):
    #     for j in range(len(inputList)):
    #         if j == len(inputList):
    #             break
    #         for i in range(len(inputList[j])):
    #             if not len(inputList[j][i]) == size:
    #                 del inputList[j]
    #     return inputList


# do = DataOrganizer()
# print(do.cutFirstTimeStep(test))
