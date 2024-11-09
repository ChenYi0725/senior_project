import tools.data_organizer
import tqdm
organizer = tools.data_organizer.DataOrganizer()
data_paths = [
    ("frontClockwiseData", "data_set_2hands/front_clockwise_2hands"),
    ("frontCounterClockwiseData", "data_set_2hands/front_counter_clockwise_2hands"),
    ("leftDownData", "data_set_2hands/left_down_2hands"),
    ("leftUpData", "data_set_2hands/left_up_2hands"),
    ("rightDownData", "data_set_2hands/right_down_2hands"),
    ("rightUpData", "data_set_2hands/right_up_2hands"),
    ("topLeftData", "data_set_2hands/top_left_2hands"),
    ("topRightData", "data_set_2hands/top_right_2hands"),
    ("stopData", "data_set_2hands/stop_2hands")
]

data_dict = {}
for name, path in tqdm(data_paths, desc="Loading Data"):
    data_dict[name] = organizer.getDataFromTxt(path)

frontClockwiseData = data_dict["frontClockwiseData"]
frontCounterClockwiseData = data_dict["frontCounterClockwiseData"]
leftDownData = data_dict["leftDownData"]
leftUpData = data_dict["leftUpData"]
rightDownData = data_dict["rightDownData"]
rightUpData = data_dict["rightUpData"]
topLeftData = data_dict["topLeftData"]
topRightData = data_dict["topRightData"]
stopData = data_dict["stopData"]