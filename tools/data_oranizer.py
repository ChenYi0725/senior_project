import numpy as np
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
# import data_set_2hands


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

        # filePath = os.path.join(os.path.dirname(__file__), "..", f"{fileName}.txt")
        # print("file path : " + filePath)
        with open(f"{fileName}.txt", "r") as file:
            content = file.read()
        result = eval(content)
        return result

    def getAccelerate(self, npArray):
        for i in range(len(npArray)):
            for j in reversed(range(len(npArray[i]))):
                for k in reversed(range(len(npArray[i][j]))):
                    if not j < 1:
                        npArray[i][j][k] = npArray[i][j][k] - npArray[i][j - 1][k]
        npArray = self.cutFirstTimeStep(npArray)
        return npArray

    def checkData(self, fileName):
        targetFile = self.getDataFromTxt(fileName)
        errorList = []
        for i in range(len(targetFile)):
            if not len(targetFile[i]) == 21:
                errorList.append(i)
                continue
            for j in range(len(targetFile[i])):
                if not len(targetFile[i][j]) == 84:
                    errorList.append(i)
                continue
        return errorList

    def reverseTimeData(self, npArray):
        npArray = [sublist[::-1] for sublist in npArray]
        return npArray
