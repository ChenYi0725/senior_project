import data_organizer as data_organizer

organizer = data_organizer.data_organizer()
num = 0  # 未達需求的資料集數量
needed_data = 200


def check_data(file_location):
    global num
    file_name = file_location.replace("data_set_2hand\\", "")
    print(f"{file_name}:", end="")
    print(organizer.find_error_data(f"{file_location}"), end="  ")
    print("len:", end="")
    if len(organizer.get_data_from_txt(f"{file_location}")) >= needed_data:
        print(len(organizer.get_data_from_txt(f"{file_location}")), end="")
        print(" done")
    else:
        print(len(organizer.get_data_from_txt(f"{file_location}")))
        num = num + 1


def clean_data(file_location, error_data):
    dataset = organizer.get_data_from_txt(f"{file_location}")
    del dataset[error_data]
    dataset = str(dataset)
    with open("tools/cleaned_data.txt", "w") as f:
        f.write(dataset)


print("start")
check_data("exhibit_2way/exhibit_data_set/horizontal")
check_data("exhibit_2way/exhibit_data_set/vertical")
check_data("exhibit_2way/exhibit_data_set/stop")
# checkData("front_view_dataset\\bright_dataset\\F'")
# checkData("front_view_dataset\\bright_dataset\\F")
# checkData("front_view_dataset\\bright_dataset\\F'")
# checkData("front_view_dataset\\bright_dataset\\L")
# checkData("front_view_dataset\\bright_dataset\\L'")
# checkData("front_view_dataset\\stop")




# checkData("data_set_2hands\\left_down_2hands")
# checkData("data_set_2hands\\left_up_2hands")
# checkData("data_set_2hands\\top_left_2hands")
# checkData("data_set_2hands\\top_right_2hands")
# checkData("data_set_2hands\\stop_2hands")
# checkData("data_set_2hands\\right_up_2hands")
# checkData("data_set_2hands\\right_down_2hands")
# checkData("data_set_2hands\\bottom_left_2hands")
# checkData("data_set_2hands\\bottom_right_2hands")
# checkData("data_set_2hands\\front_clockwise_2hands")
# checkData("data_set_2hands\\front_counter_clockwise_2hands")
# checkData("data_set_2hands\\back_clockwise_2hands")
# checkData("data_set_2hands\\back_counter_clockwise_2hands")
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
