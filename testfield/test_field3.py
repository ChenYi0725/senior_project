import numpy as np

# import tools.data_oranizer

# organizer = tools.data_oranizer.DataOrganizer()
eList = [1, 2, 5]
theList = [0, 10, 20, 3, 4, 50, 6, 7]
dnu = 0
# theList = organizer.getDataFromTxt("data_set_2hand\\bottom_right_2hands")
for i in range(len(theList)):
    if i in eList:
        del theList[i]
        i = i - 1
        d

print(theList)
