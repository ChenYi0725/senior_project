import numpy as np
import data_oranizer

organizer = data_oranizer.DataOrganizer()

newlist = []
theList = organizer.getDataFromTxt("data_set_2hand\\right_down_2hands")
print(len(theList[281]))
