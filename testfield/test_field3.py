import numpy as np

dataLengthList = [3, 4, 5, 6, 7]
del dataLengthList[-1]
target = np.zeros(25)
targetValue = 0
targetPointer = 0
print(target)
for i in dataLengthList:
    target[targetPointer : targetPointer + i] = targetValue
    targetPointer = targetPointer + i
    targetValue = targetValue + 1
print(target)
