count = 0
my_list = []


def plass1():
    global count
    count = count + 1
    if len(my_list) > 20:
        del my_list[0]
        my_list.append(count)
    else:
        my_list.append(count)
    return count, my_list
