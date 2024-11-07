import time


# 不用j
# 保留i，並將i前面的?個額外取出
def findMoveTime(previousFingertips):
    additionalReserve = 2
    threshHold = [90, 90]
    # --------------
    isMove = False
    if len(previousFingertips) > 2:
        currentFingertips = previousFingertips[-1]
        for i in range(len(previousFingertips) - 1):  # 不包含最後一個 list
            for j in range(len(currentFingertips)):
                diff = abs(currentFingertips[j] - previousFingertips[i][j])
                if diff > threshHold[j % 2]:
                    isMove = True
                    break
            if isMove:
                moveTime = i
                break
        if moveTime < additionalReserve:
            return previousFingertips
        else:
            return previousFingertips[moveTime - additionalReserve:]


    return i


start_time = time.time()
l1 = [
    [1, 1, 1, 1, 5, 8, 4, 1, 2, 5],
    [1, 2, 3, 7, 7, 7, 74, 17, 27, 57],
    [1, 2, 3, 4, 5, 8, 4, 1, 2, 5],
    [1, 2, 3, 4, 5, 8, 4, 1, 2, 5],  # move
    [1, 2, 3, 4, 5, 8, 4, 1, 2, 5],
    [1, 2, 3, 4, 5, 8, 4, 1, 2, 5],
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [55, 4, 5, 6, 72, 2, 4, 1, 7, 1],
    [8000, 4, 5, 6, 7, 6, 8, 10, 11, 13, 4],
    [2, 6, 7, 8, 9, 1, 3, 4, 5, 6],  # current
]
result = []
for i in range(len(l1) - 1):  # 不包含最後一個 list
    diff_abs = [abs(l1[-1][j] - l1[i][j]) for j in range(len(l1[-1]))]
    result.append(diff_abs)
end_time = time.time()

# max只看頭

# print(result)

# print(result.index((max(result))))
# print(max(result).index(max(max(result))))

# print(max(max(result)))
# print(l1[result.index((max(result)))][max(result).index(max(max(result)))])
# print(end_time - start_time)

# # 抓到啟動後往前抓5個

# print(l1[result.index((max(result))) - 1 : -1])

print(findMoveTime(l1))
