import numpy as np
def keepIndexFingerAndTips(inputList):
    fingerAndTips = np.array([
        0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        24, 25, 32, 33, 40, 41, 42, 43, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 66, 67, 74, 75,
        82, 83
    ])
    filtered_array = inputList[fingerAndTips]
    return filtered_array



# list1 = np.array()
# list1=[]
# for i in range(84):
#     list1.append(i*10)
# list1 = np.array(list1)
# list1 = keepIndexFingerAndTips(list1)
# print(list1)
l = [
        0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        24, 25, 32, 33, 40, 41, 42, 43, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 66, 67, 74, 75,
        82, 83
    ]
print(len(l))