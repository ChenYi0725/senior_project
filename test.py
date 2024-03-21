import numpy as np
import data_oranizer as do

organizer = do.DataOrganizer()
# print("wait a minute")
myList = organizer.getDataFromTxt("left_data")
# print("down")
# print(len(myList))
# myList = np.array(myList)

# myList = organizer.cutInhomogeneousData(myList, 42)

# del myList[249]
# del myList[249]
# del myList[249]
# del myList[249]
# del myList[249]

for j in range(len(myList)):
    if j == len(myList):
        break
    for i in range(len(myList[j])):
        if not len(myList[j][i]) == 42:
            del myList[j]
            # print("error")
            # print(j)
del myList[38]
del myList[83]
for j in range(len(myList)):
    if j == len(myList):
        break
    for i in range(len(myList[j])):
        if not len(myList[j][i]) == 42:
            del myList[j]
            print("error")
            print(j)

# myList = np.array(myList)


myListString = str(myList)
with open("clean_up.txt", "w") as f:
    f.write(myListString)
print("done")

# myList = [[[3, 3], [5, 5]], [[8, 8], [4, 1]]]
# myList = np.array(myList)
# print(myList)

# with open("left_data.txt", "r") as file:
#     content = file.read()


# my_list = eval(content)
# print(len(my_list))
