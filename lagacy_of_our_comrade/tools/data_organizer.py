import numpy as np

# from numba import jit


class DataOrganizer:
    def _cutFirstTimeStep(self, npArray):
        npArray = npArray[:, 1:, :]
        return npArray

    def _MediapipeNodeToIndex(self, inputList):
        index = []
        for i in inputList:
            index.append(i * 2)
            index.append(i * 2 + 1)
        secondHand = []
        for i in index:
            secondHand.append(i + 42)
        index.extend(secondHand)
        return index

    def _removePalmNode(self, inputList):
        palm = self._MediapipeNodeToIndex([0, 1, 5, 9, 13, 17])
        inputList = np.delete(inputList, palm, axis=2)  # 刪除對應的索引(in features)
        return inputList

    def _keepIndexFingerAndThumb(self, inputList):
        targetIndex = np.array(
            self._MediapipeNodeToIndex([0, 3, 4, 7, 8 ])
        )
        filtered_array = np.take(inputList, targetIndex, axis=2)
        return filtered_array

    def _keepIndexFingerAndTips(self, inputList):
        fingerAndTips = np.array(
            self._MediapipeNodeToIndex([0, 4, 5, 6, 7, 8, 12, 16, 20])
        )
        filtered_array = np.take(inputList, fingerAndTips, axis=2)
        return filtered_array

    # @staticmethod
    # @jit(nopython=True)
    def preprocessData(self, inputList):
        inputList = np.array(inputList)
        inputList = self._normalizedWithEachTimeSteps(inputList)
        # inputList = self._getRelativeWithFirstTimeStep(inputList)
        inputList = self._getRelativeLocation(inputList)
        # inputList = self.getAccelerate(inputList)
        inputList = self._removePalmNode(inputList)
        return inputList

    def preprocessForShirnkModel(self, inputList):
        inputList = np.array(inputList)
        inputList = self._normalizedWithEachTimeSteps(inputList)
        inputList = self._getRelativeLocation(inputList)
        inputList = self._keepIndexFingerAndTips(inputList)
        return inputList
    
    def preprocessForIndexaAndThumbModel(self, inputList):
        inputList = np.array(inputList)
        inputList = self._normalizedWithEachTimeSteps(inputList)
        inputList = self._getRelativeLocation(inputList)
        inputList = self._keepIndexFingerAndThumb(inputList)
        return inputList

    @staticmethod
    def _getRelativeLocation(npArray):  # 以各個時間步的左手腕為基準，輸入:(data number,time step, features)
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

    @staticmethod
    def _normalizedWithEachTimeSteps(
        inputList,
    ):  # 輸入:(data number,time step, features)

        for i in range(len(inputList)):
            for j in range(inputList.shape[1]):
                inputList[i, j] = (inputList[i, j] - inputList[i, j].min()) / (
                    inputList[i, j].max() - inputList[i, j].min()
                )
        return inputList

    def normalizedOneDimensionList(self, inputList):
        npInputList = np.array(inputList)
        normalizedList = (npInputList - npInputList.min()) / (
            npInputList.max() - npInputList.min()
        )
        normalizedList = normalizedList.tolist()
        return normalizedList

    def _getRelativeWithFirstTimeStep(self, npArray):    #以第一個時間步左手腕為基準
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

    def getDataFromTxt(self, fileName):     #從檔案中取得list資料
        with open(f"{fileName}.txt", "r") as file:
            content = file.read()
        result = eval(content)
        return result

    def getAccelerate(self, npArray):   # 取得與上一時間步的差距
        for i in range(len(npArray)):
            for j in reversed(range(len(npArray[i]))):
                for k in reversed(range(len(npArray[i][j]))):
                    if not j < 1:
                        npArray[i][j][k] = npArray[i][j][k] - npArray[i][j - 1][k]
        npArray = self._cutFirstTimeStep(npArray)
        return npArray

    def findErrorData(self, fileName):  #用於尋找錯誤資料
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
