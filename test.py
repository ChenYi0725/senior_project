import tools.data_organizer
import numpy as np


def getDiff(list):
    diffs = []
    for data in list:
        for ts in range(len(data) - 1):
            for i in range(len(data[ts])):
                diff = data[ts + 1][i] - data[ts][i]
                diffs.append(diff)
    return diffs


organizer = tools.data_organizer.DataOrganizer()
backClockwiseData = organizer.getDataFromTxt("data_set_2hands/back_clockwise_2hands")
backCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/back_counter_clockwise_2hands"
)
bottomLeftData = organizer.getDataFromTxt("data_set_2hands/bottom_left_2hands")
bottomRightData = organizer.getDataFromTxt("data_set_2hands/bottom_right_2hands")
frontClockwiseData = organizer.getDataFromTxt("data_set_2hands/front_clockwise_2hands")
frontCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/front_counter_clockwise_2hands"
)
leftDownData = organizer.getDataFromTxt("data_set_2hands/left_down_2hands")
leftUpData = organizer.getDataFromTxt("data_set_2hands/left_up_2hands")
rightDownData = organizer.getDataFromTxt("data_set_2hands/right_down_2hands")
rightUpData = organizer.getDataFromTxt("data_set_2hands/right_up_2hands")
topLeftData = organizer.getDataFromTxt("data_set_2hands/top_left_2hands")
topRightData = organizer.getDataFromTxt("data_set_2hands/top_right_2hands")
stopData = organizer.getDataFromTxt("data_set_2hands/stop_2hands")

backClockwiseData = organizer.getDataFromTxt("data_set_2hands/back_clockwise_2hands")
backCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/back_counter_clockwise_2hands"
)
bottomLeftData = organizer.getDataFromTxt("data_set_2hands/bottom_left_2hands")
bottomRightData = organizer.getDataFromTxt("data_set_2hands/bottom_right_2hands")
frontClockwiseData = organizer.getDataFromTxt("data_set_2hands/front_clockwise_2hands")
frontCounterClockwiseData = organizer.getDataFromTxt(
    "data_set_2hands/front_counter_clockwise_2hands"
)
print("finishLoading")

all_data = [
    backClockwiseData,
    backCounterClockwiseData,
    bottomLeftData,
    bottomRightData,
    frontClockwiseData,
    frontCounterClockwiseData,
    leftDownData,
    leftUpData,
    rightDownData,
    rightUpData,
    topLeftData,
    topRightData,
    stopData,
]
too_large_count = 0
large_count = 0

for data in all_data:
    for timesteps in data:
        for features in timesteps:
            for feature in features:
                if abs(feature) > 2:
                    print("too large")
                    too_large_count = too_large_count + 1
                elif abs(feature) > 1:
                    print("larege")
                    large_count = large_count + 1
                else:
                    pass
print(too_large_count)
print(large_count)
print("finish test")

all_diff = []
for data in all_data:
    all_diff.append(getDiff(data))


all_diff_average = np.mean(all_diff)
print(all_diff_average)
