import matplotlib.pyplot as plt
import tools.data_organizer as data_organizer

organizer = data_organizer.DataOrganizer()

data = organizer.getDataFromTxt(
    "D:\Python_project\senior_project\\data_set_2hands/back_clockwise_2hands"
)
oneData = data[100]

data_ranges = [(54, 60, "b"), (70, 76, "g"), (62, 68, "r")]

# plt.figure() 應放在循環內，每次迭代創建新的圖表
for idx, pdata in enumerate(oneData):
    plt.figure(figsize=(8, 6))

    for start, end, color in data_ranges:
        segment = pdata[start:end]
        x_values = segment[::2]  # 奇數當x
        y_values = segment[1::2]  # 偶數當y
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="",
            color=color,
            label=f"Range {start}-{end}",
        )

    plt.title(f"Coordinate Plot for Data {idx+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()
