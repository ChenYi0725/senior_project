import numpy as np
data = [1,2,3,4,5,6,7,11,22,33,44,55,66,77,88]
dataLengthList = [7,8]
labels = np.zeros(len(data), dtype=np.int32)  # total
labelsPointer = 0
labelsValue = 0
for i in dataLengthList:
    labels[labelsPointer : labelsPointer + i] = labelsValue
    labelsPointer = labelsPointer + i
    labelsValue = labelsValue + 1
# =========================
print(f"label:{labels}")
print(f"label:{labels.shape}")

errorList = np.zeros(len(data))
print(f"label:{errorList}")
print(f"label:{errorList.shape}")


# data shape: (13000, 21, 60)
# labels shape: (13000,)
# inputLength shape: (13000,)
# labelLength shape: (13000,)
# inputLength example: [21 21 21 21 21 21 21 21 21 21]
# labelLength example: [1 1 1 1 1 1 1 1 1 1]