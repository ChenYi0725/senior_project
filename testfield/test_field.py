import numpy as np

l1 = [[[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], [[2, 3, 4, 5, 6], [5, 6, 7, 8, 9]]]
l2 = [[[2, 3, 4, 5, 6], [5, 6, 7, 8, 9]], [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]]

all_l = [l1,l2]

def getDiff(list):
    diffs = []
    for data in list:
        for ts in range(len(data)-1):
            for i in range(len(data[ts])):
                diff = data[ts+1][i] - data[ts][i]
                diffs.append(diff)
    return diffs

diffs_l1 = getDiff(l1)
diffs_l2 = getDiff(l2)

all_data = [l1,l2]
all_diff = []
for data in all_data:
    all_diff.append(getDiff(data))

all_diff_average = np.mean(all_diff)
all_data_np = np.array(all_data)
average_data = np.mean(all_data_np)
print(all_diff_average)

# average_diff_l1 = np.mean(diffs_l1, axis=0)
# average_diff_l2 = np.mean(diffs_l2, axis=0)

# print("l1中最裡面子list的差值平均值:", average_diff_l1)
# print("l2中最裡面子list的差值平均值:", average_diff_l2)
