import test_field4
import numpy as np

def normalizedOneDimensionList(inputList):
    npInputList = np.array(inputList)
    normalizedList = (npInputList - npInputList.min()) / (npInputList.max() - npInputList.min())
    normalizedList= normalizedList.tolist()
    return normalizedList
list = [1,2]
target = [1,2,3,4,5,6,7,8,9,0,10]
target2 = [1,2,3,4,5,6,7,8,9,0,10]
# target = normalizedOneDimensionList(target)
target3 = target +target2
# for i in range(len(target)):
#     target[i] = target[i] - list[i%2]
# print(target3)

def my_function():
    if not hasattr(my_function, 'counter'):
        my_function.counter = 0  # 初始化靜態變數
    my_function.counter += 1
    print("Counter is:", my_function.counter)

# 正常使用
my_function()  # Output: Counter is: 1
my_function()  # Output: Counter is: 2

# 在其他地方更改 my_function 的靜態變數
my_function.counter = 100  # 更改靜態變數

# 再次調用 my_function
my_function()  # Output: Counter is: 101
