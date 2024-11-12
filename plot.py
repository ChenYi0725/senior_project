import tools.data_organizer
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

#A 左手 B 右手
def draw_plot(total_data):

    for data in total_data:
        total_A_x,total_A_y = [],[]
        total_B_x,total_B_y = [],[]
        for time in data:
            x_values_A, y_values_A = [], []
            x_values_B, y_values_B = [], []
            for i in range(0, len(time)):  
                # if lock:
                    if i == 16:
                        x_values_A.append(time[i])
                        total_A_x.append(time[i])
                    elif i == 17:
                        y_values_A.append(time[i])
                        total_A_y.append(time[i])
                    elif i == 58:
                        x_values_B.append(time[i])
                        total_B_x.append(time[i])
                    elif i == 59:
                        y_values_B.append(time[i])
                        total_B_y.append(time[i])

            plt.scatter(x_values_A, y_values_A, marker='o', label='Group A (16, 17)')
            
            plt.scatter(x_values_B, y_values_B, marker='x', label='Group B (58, 59)')
            
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.title("Accelerate by Group")
            # plt.legend()
            # plt.grid(True)
            # plt.show()
        for x,y in zip(total_A_x,total_A_y):
            print(f"({round(10*x, 4)},{round(10*y, 4)})")
        print("====================")
        for x,y in zip(total_B_x,total_B_y):
            print(f"({round(10*x, 4)},{round(10*y, 4)})") 
        print("====================")




organizer = tools.data_organizer.DataOrganizer()
# data_paths = [
#     ("frontClockwiseData", "data_set_2hands/front_clockwise_2hands"),
#     ("frontCounterClockwiseData", "data_set_2hands/front_counter_clockwise_2hands"),
#     ("leftDownData", "data_set_2hands/left_down_2hands"),
#     ("leftUpData", "data_set_2hands/left_up_2hands"),
#     ("rightDownData", "data_set_2hands/right_down_2hands"),
#     ("rightUpData", "data_set_2hands/right_up_2hands"),
#     ("topLeftData", "data_set_2hands/top_left_2hands"),
#     ("topRightData", "data_set_2hands/top_right_2hands"),
#     ("stopData", "data_set_2hands/stop_2hands")
# ]
# data_dict = {}
# for name, path in tqdm(data_paths, desc="Loading Data"):
#     temp_data = organizer.getDataFromTxt(path)
#     temp_data = np.array(temp_data)
#     temp_data = organizer.getAccelerate(temp_data)
#     # draw_plot(temp_data)
#     data_dict[name] = temp_data

data = organizer
temp_data = organizer.getDataFromTxt("test")
temp_data = np.array(temp_data)
temp_data = organizer.getAccelerate(temp_data)
draw_plot(temp_data)

# frontClockwiseData = data_dict["frontClockwiseData"]
# frontCounterClockwiseData = data_dict["frontCounterClockwiseData"]
# leftDownData = data_dict["leftDownData"]
# leftUpData = data_dict["leftUpData"]
# rightDownData = data_dict["rightDownData"]
# rightUpData = data_dict["rightUpData"]
# topLeftData = data_dict["topLeftData"]
# topRightData = data_dict["topRightData"]
# stopData = data_dict["stopData"]

# concatenate_data = np.concatenate(
#     (
#         frontClockwiseData,
#         frontCounterClockwiseData,
#         leftDownData,
#         leftUpData,
#         rightDownData,
#         rightUpData,
#         topLeftData,
#         topRightData,
#         stopData,
#     ),
#     axis=0,
# )
# draw_plot(topLeftData)
# draw_plot(topRightData)
# draw_plot(stopData)
# draw_plot(concatenate_data)