import data_organizer as data_organizer

organizer = data_organizer.DataOrganizer()
num = 0  # 未達需求的資料集數量
neededData = 200


def checkData(fileLocation):
    global num
    fileName = fileLocation.replace("data_set_2hand\\", "")
    print(f"{fileName}:", end="")
    print(organizer.findErrorData(f"{fileLocation}"), end="  ")
    print("len:", end="")
    if len(organizer.getDataFromTxt(f"{fileLocation}")) >= neededData:
        print(len(organizer.getDataFromTxt(f"{fileLocation}")), end="")
        print(" done")
    else:
        print(len(organizer.getDataFromTxt(f"{fileLocation}")))
        num = num + 1


def cleanData(fileLocation, errorData):
    dataset = organizer.getDataFromTxt(f"{fileLocation}")
    del dataset[errorData]
    dataset = str(dataset)
    with open("tools/cleaned_data.txt", "w") as f:
        f.write(dataset)


print("start")

checkData("data_set_2hands\\left_down_2hands")
checkData("data_set_2hands\\left_up_2hands")
checkData("data_set_2hands\\top_left_2hands")
checkData("data_set_2hands\\top_right_2hands")
checkData("data_set_2hands\\stop_2hands")
checkData("data_set_2hands\\right_up_2hands")
checkData("data_set_2hands\\right_down_2hands")
checkData("data_set_2hands\\bottom_left_2hands")
checkData("data_set_2hands\\bottom_right_2hands")
checkData("data_set_2hands\\front_clockwise_2hands")
checkData("data_set_2hands\\front_counter_clockwise_2hands")
checkData("data_set_2hands\\back_clockwise_2hands")
checkData("data_set_2hands\\back_counter_clockwise_2hands")
# cleanData("test_data_set\\stop_test",142)
# ----------------------------------------------
# checkData("test_data_set\\stop_test")
# checkData("test_data_set\\l_test")
# checkData("test_data_set\\l'_test")
# checkData("test_data_set\\r_test")
# checkData("test_data_set\\r'_test")
# checkData("test_data_set\\u_test")
# checkData("test_data_set\\u'_test")
# checkData("test_data_set\\d_test")
# checkData("test_data_set\\d'_test")
# checkData("test_data_set\\f_test")
# checkData("test_data_set\\f'_test")
# checkData("test_data_set\\b_test")
# checkData("test_data_set\\b'_test")
print("")
print(f"unfinished {num}")
