import matplotlib.pyplot as plt
import data_oranizer

organizer = data_oranizer.DataOrganizer()
data = organizer.getDataFromTxt("D:\Python_project\senior_project\\result")
pdata = data[0][0]  # 取最後一個
pdata = pdata[42:]
x_values = pdata[::2]  # 將列表中的奇數索引視為 x 坐標
y_values = pdata[1::2]  # 將列表中的偶數索引視為 y 坐標
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, marker="o", linestyle="", color="b")
plt.title("Coordinate Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
