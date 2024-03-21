from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class HandGraph:
    def __init__(self, data, index):
        self.dataX = [item[1] for item in data]
        self.dataY = [item[2] for item in data]
        self.dataName = [str(item[0]) for item in data]
        self.graph = plt.figure(figsize=(8, 6)).add_subplot(index)
        self.graph.plot(self.dataX, self.dataY, marker="o", linestyle="")
        self._setGraph()
        self._setName()

    def _setGraph(self):
        self.graph.set_xlabel("X")
        self.graph.set_ylabel("Y")
        self.graph.set_title("hand")
        self.graph.grid(True)

    def _setName(self):
        for i, name in enumerate(self.dataName):
            self.graph.annotate(
                name,
                (self.dataX[i], self.dataY[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )


def rotate(data, degree):
    dataMatrix = np.array(data)[:, 1:]
    theta = np.deg2rad(degree)
    rotationMatrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotatedData = np.dot(dataMatrix, rotationMatrix)
    rotatedData = np.array([[i] + row.tolist() for i, row in enumerate(rotatedData)])
    return rotatedData


def preprocessData(data):
    origin = data[0][:]
    for item in data:
        item[1] = item[1] - origin[1]
        item[2] = item[2] - origin[2]
    return data


data = [
    [0, 261, 376],
    [1, 313, 363],
    [2, 348, 330],
    [3, 371, 300],
    [4, 397, 282],
    [5, 314, 269],
    [6, 331, 221],
    [7, 340, 190],
    [8, 346, 166],
    [9, 282, 263],
    [10, 288, 208],
    [11, 291, 174],
    [12, 292, 147],
    [13, 252, 267],
    [14, 251, 218],
    [15, 252, 187],
    [16, 252, 161],
    [17, 222, 277],
    [18, 208, 239],
    [19, 201, 213],
    [20, 195, 188],
]

# data2 = [
#     [0, 156, 360],
#     [1, 181, 413],
#     [2, 224, 441],
#     [3, 263, 453],
#     [4, 296, 466],
#     [5, 280, 397],
#     [6, 332, 409],
#     [7, 366, 417],
#     [8, 395, 422],
#     [9, 279, 366],
#     [10, 340, 365],
#     [11, 379, 366],
#     [12, 413, 367],
#     [13, 269, 338],
#     [14, 323, 332],
#     [15, 360, 332],
#     [16, 392, 333],
#     [17, 252, 311],
#     [18, 292, 297],
#     [19, 320, 294],
#     [20, 347, 292],
# ]
data = preprocessData(data)

data2 = rotate(data, 30)  # 旋轉30度
print(data2)

handGraph1 = HandGraph(data=data, index=111)
plt.gca().invert_yaxis()

handGraph2 = HandGraph(data=data2, index=111)
plt.gca().invert_yaxis()

plt.show()
