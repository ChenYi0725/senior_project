import data_oranizer

organizer = data_oranizer.DataOrganizer()
num = 0


def checkData(fileLocation):
    global num
    fileName = fileLocation.replace("data_set_2hand\\", "")
    print(f"{fileName}:", end="")
    print(organizer.checkData(f"{fileLocation}"), end="  ")
    print("len:", end="")
    if len(organizer.getDataFromTxt(f"{fileLocation}")) >= 300:
        print(len(organizer.getDataFromTxt(f"{fileLocation}")), end="")
        print(" done")
    else:
        print(len(organizer.getDataFromTxt(f"{fileLocation}")))
        num = num + 1


checkData("data_set_2hand\\left_down_2hands")
checkData("data_set_2hand\\left_up_2hands")
checkData("data_set_2hand\\top_left_2hands")
checkData("data_set_2hand\\top_right_2hands")
checkData("data_set_2hand\\stop_2hands")
checkData("data_set_2hand\\right_up_2hands")
checkData("data_set_2hand\\right_down_2hands")
checkData("data_set_2hand\\bottom_left_2hands")
checkData("data_set_2hand\\bottom_right_2hands")
checkData("data_set_2hand\\front_clockwise_2hands")
checkData("data_set_2hand\\front_counter_clockwise_2hands")
checkData("data_set_2hand\\back_clockwise_2hands")
checkData("data_set_2hand\\back_counter_clockwise_2hands")

print("")
print(f"still need {num}")
