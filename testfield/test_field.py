import numpy as np

# import tools.data_organizer as do

input_list = []

for i in range(84):
    input_list.append(i)


def expandTo2Hands(fingerlist):
    for i in range(len(fingerlist)):
        newFeature = fingerlist[i] + 42
        fingerlist.append(newFeature)
    return fingerlist


def removePalmNode(inputList):
    palm = [0, 1, 2, 3, 10, 11, 18, 19, 26, 27, 34, 35]
    palm = expandTo2Hands(palm)
    print(palm)
    palm.sort(reverse=True)
    for i in palm:
        del inputList[i]
    return inputList


def npPalmRemove(inputList):
    inputList = np.array(inputList)
    palm = [
        0,
        1,
        2,
        3,
        10,
        11,
        18,
        19,
        26,
        27,
        34,
        35,
        42,
        43,
        44,
        45,
        52,
        53,
        60,
        61,
        68,
        69,
        76,
        77,
    ]
    inputList = np.delete(inputList, palm, axis=0)  # 刪除對應的索引
    return inputList


print(npPalmRemove(inputList=input_list).shape)
